"""Build an :class:`LLMClient` for Gemma 4 31B from :class:`LLMConfig`, and test the connection."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import yaml
from pydantic import BaseModel, ValidationError

from ..config import DEFAULT_MODEL, LLMConfig
from .base import ImagePart, LLMClient, LLMError, Message, MockBackend, schema_for
from .gemini import GEMINI_BASE_URL, GEMINI_GEMMA_MODEL, GeminiBackend
from .jsonutil import extract_json
from .openai_compat import OpenAICompatBackend, redact_url

log = logging.getLogger(__name__)

# Default endpoints of the OpenAI-compatible servers (used when no base_url was configured).
OPENAI_COMPAT_DEFAULT_URLS: dict[str, str] = {
    "openai": "http://localhost:8000/v1",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    "ollama": "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "lmstudio": "http://localhost:1234/v1",
}
# Gemma 4 31B model names where the server's naming differs from the Hugging Face id.
DEFAULT_MODEL_NAMES: dict[str, str] = {"ollama": "gemma4:31b", "gemini": GEMINI_GEMMA_MODEL}
_ALIASES = {"llama.cpp": "llamacpp", "llama-cpp": "llamacpp", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
            "openai-compatible": "openai", "openai_compat": "openai", "google": "gemini"}
BACKENDS = tuple(OPENAI_COMPAT_DEFAULT_URLS) + ("gemini", "mock", "none")

Responder = Callable[[Sequence[Message], Optional[dict], str], Any]


def _backend_name(cfg: LLMConfig) -> str:
    name = (cfg.backend or "none").strip().lower()
    return _ALIASES.get(name, name)


def base_url_is_explicit(cfg: LLMConfig) -> bool:
    """Whether ``cfg.base_url`` was configured by the user rather than left at LLMConfig's default.

    ``LLMConfig.base_url`` defaults to a concrete URL, so an explicitly configured
    ``http://localhost:8000/v1`` cannot be told apart by value. Resolution: a
    ``base_url_explicit`` attribute set by the caller; a value different from the
    default; ``$SHIPPID_LLM_BASE_URL``; ``llm.base_url`` in the config file that
    :func:`load_config` would read (``$SHIPPID_CONFIG`` or ``./shippid.yaml``).
    """
    flag = getattr(cfg, "base_url_explicit", None)
    if flag is not None:
        return bool(flag)
    if not cfg.base_url:
        return False
    if cfg.base_url != LLMConfig().base_url or os.environ.get("SHIPPID_LLM_BASE_URL"):
        return True
    for cand in (os.environ.get("SHIPPID_CONFIG"), "shippid.yaml"):
        if cand and Path(cand).is_file():  # load_config uses the first file it finds
            try:
                data = yaml.safe_load(Path(cand).read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                return False
            llm = data.get("llm") if isinstance(data, dict) else None
            return isinstance(llm, dict) and bool(llm.get("base_url"))
    return False


def make_client(cfg: LLMConfig, responder: Optional[Responder] = None, *,
                base_url_explicit: Optional[bool] = None) -> Optional[LLMClient]:
    """LLM client for ``cfg.backend``; ``None`` when the LLM is disabled (``none``).

    Without a configured ``base_url`` (see :func:`base_url_is_explicit`; override with
    ``base_url_explicit``) the backend's own default endpoint is used, e.g. port
    11434 for Ollama. ``mock`` needs an explicit ``responder`` (tests / dry runs); its
    output is labelled ``mock`` in usage records and never passes for Gemma output.
    """
    name = _backend_name(cfg)
    if name in ("none", ""):
        return None
    explicit = base_url_is_explicit(cfg) if base_url_explicit is None else bool(base_url_explicit)
    model = cfg.model or DEFAULT_MODEL
    if model == DEFAULT_MODEL and name in DEFAULT_MODEL_NAMES:
        model = DEFAULT_MODEL_NAMES[name]
    json_mode = cfg.json_mode if cfg.json_mode in ("json_schema", "json_object", "prompt") else "json_schema"
    if name in OPENAI_COMPAT_DEFAULT_URLS:
        base_url = cfg.base_url if explicit else OPENAI_COMPAT_DEFAULT_URLS[name]
        extra = dict(cfg.extra_body or {})
        top_k = extra.pop("top_k", None)
        backend = OpenAICompatBackend(
            model, base_url, api_key=cfg.api_key, timeout=cfg.timeout, max_retries=cfg.max_retries, top_k=top_k,
            extra_body=extra, name=name,
            response_format="json_object" if json_mode == "json_object" else "json_schema",
            thinking_default=cfg.enable_thinking,
        )
    elif name == "gemini":
        base_url = cfg.base_url if explicit else GEMINI_BASE_URL
        backend = GeminiBackend(model, api_key=cfg.api_key, base_url=base_url, timeout=cfg.timeout,
                                max_retries=cfg.max_retries,
                                json_mode="json_mime" if json_mode == "json_object" else "json_schema",
                                thinking_default=cfg.enable_thinking)
    elif name == "mock":
        if responder is None:
            raise ValueError("backend 'mock' needs a responder (make_client(cfg, responder=...)); "
                             "it is for tests only and does not run Gemma")
        backend = MockBackend(responder, model=model if model != DEFAULT_MODEL else "mock")
    else:
        raise ValueError(f"unknown LLM backend {cfg.backend!r}; expected one of {', '.join(BACKENDS)}")
    if name != "mock":
        backend.base_url_source = "configured" if explicit else "backend default"  # type: ignore[attr-defined]
        if not explicit:
            log.info("no base_url configured; using the %s default endpoint %s", name, redact_url(base_url))
    # json_object is realised by the backend's response format; the client keeps passing the schema
    client_mode = "prompt" if json_mode == "prompt" else "json_schema"
    context = getattr(cfg, "context_tokens", None)
    return LLMClient(backend, temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens,
                     json_mode=client_mode, cache_dir=cfg.cache_dir,
                     context_tokens=int(context) if isinstance(context, (int, float)) and context > 0 else None)


# --------------------------------------------------------------------------- connection check


class _Probe(BaseModel):
    shape: str
    color: str
    text: str = ""


_SHAPE_WORDS = ("circle", "disc", "disk", "round", "dot", "ring", "원")
_COLOR_WORDS = ("red", "빨", "적")


def probe_image() -> bytes:
    """A 96x96 PNG: a filled red circle with the letters 'P1' below it, on white."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=96, height=96)
    page.draw_circle((48, 40), 26, color=(0.85, 0, 0), fill=(0.85, 0, 0))
    page.insert_text((38, 88), "P1", fontsize=16, fontname="hebo", color=(0, 0, 0))
    png = page.get_pixmap(alpha=False).tobytes("png")
    doc.close()
    return png


def _display_url(backend: Any) -> Optional[str]:
    """The backend's endpoint without credentials (user info / query), or None."""
    shown = getattr(backend, "display_url", None)
    if isinstance(shown, str):
        return shown
    url = getattr(backend, "url", None)
    return redact_url(url) if url else None


def check_connection(client: LLMClient, *, max_tokens: int = 256) -> dict[str, Any]:
    """Send one tiny text + image JSON request straight to the backend (bypassing the cache).

    Reports reachability, latency, the model name the server answered with,
    whether structured JSON output works (and in which mode after any
    fallback), whether the image was understood, and whether reasoning came back.
    """
    backend = client.backend
    report: dict[str, Any] = {
        "ok": False, "backend": backend.name, "model": backend.model, "server_model": None,
        "url": _display_url(backend), "base_url_source": getattr(backend, "base_url_source", None),
        "latency_s": None, "json_valid": False,
        "json_schema": False, "structured_output": None, "images": False, "reasoning": False,
        "prompt_tokens": 0, "completion_tokens": 0, "answer": None, "error": None,
    }
    schema = schema_for(_Probe)
    messages = [
        Message("system", "You are a vision test. Answer in JSON only."),
        Message("user", [ImagePart(probe_image(), label="test image"),
                         'What shape is drawn, what colour is it, and which text is written under it? Answer as '
                         'JSON {"shape": "...", "color": "...", "text": "..."}.']),
    ]
    t0 = time.time()
    try:
        resp = backend.complete(messages, json_schema=schema, schema_name="probe", temperature=0.0,
                                max_tokens=max_tokens, thinking=False)
    except LLMError as exc:
        report["latency_s"] = round(time.time() - t0, 3)
        report["error"] = backend.redact(str(exc))
        return report
    report.update(
        ok=True, latency_s=round(resp.seconds or (time.time() - t0), 3), server_model=resp.model or None,
        reasoning=bool(resp.reasoning), prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
    )
    mode = getattr(backend, "structured_output", "json_schema" if backend.supports_json_schema else "prompt")
    report["structured_output"] = mode
    report["json_schema"] = mode == "json_schema"
    try:
        probe = _Probe.model_validate(extract_json(resp.text))
    except (ValueError, ValidationError) as exc:
        report["error"] = backend.redact(f"answer is not the requested JSON: {str(exc)[:200]} (raw: {resp.text[:200]!r})")
        report["answer"] = resp.text[:500]
        return report
    report["json_valid"] = True
    report["answer"] = probe.model_dump()
    shape, color = probe.shape.lower(), probe.color.lower()
    report["images"] = any(w in shape for w in _SHAPE_WORDS) and any(w in color for w in _COLOR_WORDS)
    return report
