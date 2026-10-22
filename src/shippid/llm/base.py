"""Backend-agnostic LLM interface used to drive Gemma 4 31B.

The pipeline never talks to an HTTP API directly; it builds :class:`Message`
lists (text + PNG images) and asks :class:`LLMClient` for either free text or a
pydantic-validated JSON object. The client adds:

* JSON-schema constrained decoding when the backend supports it, otherwise a
  schema-in-prompt fallback;
* validation with retry -- the validation error is sent back to the model so it
  can repair its own output;
* an on-disk request/response cache (replay), so long analyses can be resumed
  and repeated runs are reproducible;
* token / latency accounting surfaced in reports (:class:`shippid.model.LLMUsage`);
* an answer cut off by ``max_tokens`` (``finish_reason`` "length" / "MAX_TOKENS") is repaired
  without echoing the fragment: the feedback says it was cut off, thinking is switched off for
  the retry and ``max_tokens`` is raised within ``context_tokens`` when that is known.

The replay-cache key covers the effective request: messages, schema, temperature, top_p,
max_tokens, the thinking switch resolved against the backend default, and the backend's
:meth:`LLMBackend.cache_fingerprint` (server URL, top_k / extra_body, structured-output mode).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from ..model import LLMUsage
from .budget import messages_tokens
from .jsonutil import extract_json, strip_thinking

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ImagePart:
    """A PNG (or JPEG) image passed to a multimodal model."""

    data: bytes
    mime: str = "image/png"
    label: Optional[str] = None  # caption placed before the image in the prompt

    @classmethod
    def from_file(cls, path: str | Path, label: Optional[str] = None) -> "ImagePart":
        p = Path(path)
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return cls(p.read_bytes(), mime, label)

    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64()}"


ContentPart = Union[str, ImagePart]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: Union[str, list[ContentPart]]

    def parts(self) -> list[ContentPart]:
        return [self.content] if isinstance(self.content, str) else list(self.content)

    def text(self) -> str:
        return "\n".join(p for p in self.parts() if isinstance(p, str))

    def images(self) -> list[ImagePart]:
        return [p for p in self.parts() if isinstance(p, ImagePart)]


@dataclass
class LLMResponse:
    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    model: str = ""
    finish_reason: Optional[str] = None
    raw: Any = None
    cached: bool = False


class LLMError(RuntimeError):
    pass


class LLMBackend(ABC):
    """One concrete transport (OpenAI-compatible server, Gemini API, mock...)."""

    name: str = "base"
    supports_images: bool = True
    supports_json_schema: bool = True

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
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
        """Run one chat completion.

        ``json_schema`` requests constrained JSON output (guided decoding); ``thinking``
        toggles Gemma 4's thinking mode (None = server default). Backends must place
        image parts before the text of the same message (Gemma training convention).
        """

    def cache_fingerprint(self) -> dict[str, Any]:
        """Backend settings that change the answer beyond the client's own (part of the replay-cache
        key): e.g. server URL, top_k / extra_body, structured-output mode. Empty by default."""
        return {}

    def redact(self, text: Any) -> str:
        """``text`` with this backend's credentials masked (backends holding an API key override this)."""
        return str(text)


def schema_for(model: Type[BaseModel]) -> dict[str, Any]:
    """JSON schema of a pydantic model, inlined (no $ref) for picky servers."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**inline(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)


class LLMClient:
    """High-level client: JSON validation + retry, cache, usage accounting. Thread-safe."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        json_mode: str = "json_schema",
        cache_dir: Optional[str | Path] = None,
        max_attempts: int = 3,
        schema_in_prompt: bool = True,
        context_tokens: Optional[int] = None,
    ):
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.json_mode = json_mode if backend.supports_json_schema else "prompt"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        # Guided decoding constrains the output but the model never sees the schema
        # (vLLM recipe note for Gemma 4), so field semantics also go into the prompt.
        self.schema_in_prompt = schema_in_prompt
        # server context (prompt + max_tokens), e.g. vLLM --max-model-len; None = unknown. Used to
        # raise max_tokens for the retry of a cut-off answer and by callers that size prompts.
        self.context_tokens = int(context_tokens) if context_tokens else None
        self.usage = LLMUsage(backend=backend.name, model=backend.model)
        self._lock = threading.Lock()

    # -- public ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self.backend.model

    def generate_text(self, messages: Sequence[Message], *, max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None, thinking: Optional[bool] = None) -> str:
        resp = self._call(messages, None, "text", temperature, max_tokens, thinking)
        return strip_thinking(resp.text).strip()

    def generate_json(
        self,
        messages: Sequence[Message],
        output_model: Type[T],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        validator: Optional[Callable[[T], Optional[str]]] = None,
        thinking: Optional[bool] = None,
    ) -> T:
        """Ask for an object matching ``output_model``; retries with error feedback.

        ``validator`` may return an error string for semantically invalid output
        (e.g. unknown object ids); that also triggers a repair round.
        """
        schema = schema_for(output_model)
        describe = self.schema_in_prompt or self.json_mode != "json_schema"
        msgs = _with_schema_instructions(list(messages), schema) if describe else list(messages)
        last_err = ""
        call_tokens = max_tokens or self.max_tokens
        call_thinking = thinking
        for attempt in range(1, self.max_attempts + 1):
            resp = self._call(msgs, schema if self.json_mode == "json_schema" else None,
                              output_model.__name__, temperature, call_tokens, call_thinking)
            truncated = (resp.finish_reason or "").lower() in TRUNCATED_FINISH
            try:
                data = extract_json(resp.text)
                obj = output_model.model_validate(data)
                problem = validator(obj) if validator else None
                if not problem:
                    return obj
                last_err = problem
            except (ValueError, ValidationError) as exc:
                last_err = str(exc)[:2000]
            if truncated:
                last_err = (f"the answer was cut off at the max_tokens limit ({call_tokens} tokens, finish_reason="
                            f"{resp.finish_reason}) before the JSON object was complete ({last_err[:160]})")
            log.warning("LLM output invalid (attempt %d/%d): %s", attempt, self.max_attempts, last_err[:300])
            with self._lock:
                self.usage.errors.append(f"{output_model.__name__}: {last_err[:200]}")
            if truncated:
                # never echo the cut-off fragment; give the retry more room instead of the same budget
                msgs = _with_note(list(messages), TRUNCATED_NOTE.format(tokens=call_tokens))
                if call_thinking is True or (call_thinking is None and getattr(self.backend, "thinking_default", None)):
                    call_thinking = False  # thinking shares max_tokens with the answer
                call_tokens = self._retry_tokens(msgs, call_tokens, schema if describe else None)
            else:
                msgs = list(messages) + [
                    Message("assistant", resp.text[-6000:]),
                    Message("user", "Your previous answer was not valid. Problem:\n" + last_err +
                            "\nReturn ONLY the corrected JSON object that satisfies the schema."),
                ]
            if describe:
                msgs = _with_schema_instructions(msgs, schema)
        raise LLMError(f"{output_model.__name__}: no valid output after {self.max_attempts} attempts: {last_err}")

    def _retry_tokens(self, msgs: Sequence[Message], used: int, schema: Optional[dict]) -> int:
        """``max_tokens`` for the retry of a cut-off answer: doubled, within the server context when known."""
        if not self.context_tokens:
            return used
        room = self.context_tokens - messages_tokens(msgs, schema) - 256
        return max(used, min(2 * used, room))

    # -- internals ---------------------------------------------------------------
    def _cache_key(self, messages: Sequence[Message], schema: Optional[dict], temperature: float,
                   max_tokens: int, thinking: Optional[bool]) -> str:
        """Hash of the effective request (see the module docstring)."""
        h = hashlib.sha256()
        effective = thinking if thinking is not None else getattr(self.backend, "thinking_default", None)
        fingerprint = getattr(self.backend, "cache_fingerprint", None)
        fp = fingerprint() if callable(fingerprint) else {}
        h.update(json.dumps([self.backend.name, self.backend.model, temperature, max_tokens, schema, effective,
                             self.top_p, fp], sort_keys=True, default=str).encode())
        for m in messages:
            h.update(m.role.encode())
            for p in m.parts():
                h.update(p.encode() if isinstance(p, str) else hashlib.sha256(p.data).digest())
        return h.hexdigest()

    def _call(self, messages: Sequence[Message], schema: Optional[dict], schema_name: str,
              temperature: Optional[float], max_tokens: Optional[int],
              thinking: Optional[bool] = None) -> LLMResponse:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.max_tokens
        key = self._cache_key(messages, schema, temperature, max_tokens, thinking) if self.cache_dir else None
        if key:
            f = self.cache_dir / f"{key}.json"
            if f.is_file():
                d = json.loads(f.read_text(encoding="utf-8"))
                return LLMResponse(text=d["text"], reasoning=d.get("reasoning", ""), model=d.get("model", ""),
                                   finish_reason=d.get("finish_reason"), cached=True)
        t0 = time.time()
        resp = self.backend.complete(messages, json_schema=schema, schema_name=schema_name,
                                     temperature=temperature, top_p=self.top_p, max_tokens=max_tokens,
                                     thinking=thinking)
        resp.seconds = resp.seconds or (time.time() - t0)
        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += resp.prompt_tokens
            self.usage.completion_tokens += resp.completion_tokens
            self.usage.seconds += resp.seconds
        if key:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps({"text": resp.text, "reasoning": resp.reasoning, "model": resp.model,
                            "finish_reason": resp.finish_reason}, ensure_ascii=False),
                encoding="utf-8")
        return resp


TRUNCATED_FINISH = frozenset({"length", "max_tokens"})  # OpenAI-compatible servers / Gemini API
TRUNCATED_NOTE = ("\n\nNOTE: a previous answer to this request was cut off at the max_tokens limit ({tokens} tokens) "
                  "before the JSON object was complete. Return the complete JSON object only; keep every free-text "
                  "field (reason, description, rationale, notes) short.")


def _with_note(messages: list[Message], note: str) -> list[Message]:
    """Append ``note`` to the last user turn (roles keep alternating for Gemma's chat template)."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            out[i] = Message("user", out[i].parts() + [note])
            return out
    return out + [Message("user", note.strip())]


def _with_schema_instructions(messages: list[Message], schema: dict[str, Any]) -> list[Message]:
    note = ("\n\nRespond with a single JSON object only (no prose, no markdown fences) that validates "
            "against this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False))
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            parts = out[i].parts()
            out[i] = Message("user", parts + [note])
            break
    return out


class MockBackend(LLMBackend):
    """Deterministic stand-in for tests and offline dry runs.

    ``responder(messages, json_schema, schema_name)`` returns the response text
    (or a dict/list, which is JSON-encoded). It is NOT a model: anything produced
    through it must be reported as mock output, never as Gemma output.
    """

    name = "mock"

    def __init__(self, responder: Callable[[Sequence[Message], Optional[dict], str], Any], model: str = "mock"):
        super().__init__(model)
        self.responder = responder
        self.calls: list[tuple[list[Message], Optional[dict], str]] = []

    def complete(self, messages, *, json_schema=None, schema_name="output", temperature=0.2, top_p=0.95,
                 max_tokens=4096, thinking=None) -> LLMResponse:
        self.calls.append((list(messages), json_schema, schema_name))
        out = self.responder(messages, json_schema, schema_name)
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return LLMResponse(text=text, model=self.model, prompt_tokens=sum(len(m.text()) // 4 for m in messages),
                           completion_tokens=len(text) // 4)
