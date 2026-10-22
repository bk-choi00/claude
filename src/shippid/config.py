"""Runtime configuration.

Values are resolved in this order: explicit arguments > environment variables >
``shippid.yaml`` (current directory or ``$SHIPPID_CONFIG``) > defaults.

Environment variables::

    SHIPPID_LLM_BACKEND    openai | gemini | ollama | mock | none   (default: openai)
    SHIPPID_LLM_BASE_URL   e.g. http://localhost:8000/v1 (vLLM / SGLang / LM Studio / llama.cpp)
    SHIPPID_LLM_MODEL      e.g. google/gemma-4-31B-it
    SHIPPID_LLM_API_KEY    API key (Gemini API key, or the token your gateway expects)
    SHIPPID_LLM_TIMEOUT    request timeout in seconds
    SHIPPID_LLM_CACHE_DIR  directory for the request/response cache (replay)
    SHIPPID_LLM_CONTEXT_TOKENS  server context in tokens (vLLM --max-model-len), default 32768
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_MODEL = "google/gemma-4-31B-it"


@dataclass
class LLMConfig:
    backend: str = "openai"
    base_url: str = "http://localhost:8000/v1"
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    timeout: float = 600.0
    max_retries: int = 2
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 8192
    json_mode: str = "json_schema"  # json_schema | json_object | prompt
    enable_thinking: bool = False
    max_image_side: int = 1536  # px; page renders/crops are downscaled to this
    concurrency: int = 4
    cache_dir: Optional[str] = None
    extra_body: dict[str, Any] = field(default_factory=dict)  # passed through to the server
    # server context window (prompt + max_tokens), e.g. vLLM --max-model-len; review and 3D planner
    # prompts and max_tokens are sized to fit it
    context_tokens: int = 32768

    @property
    def enabled(self) -> bool:
        return self.backend not in ("none", "", None)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    # extraction tolerances (PDF points)
    snap_tol: float = 2.5
    symbol_max_size: float = 120.0
    # 3D
    model3d_scale: float = 0.02  # metres per PDF point when mapping the drawing plane

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENV_MAP = {
    "SHIPPID_LLM_BACKEND": ("backend", str),
    "SHIPPID_LLM_BASE_URL": ("base_url", str),
    "SHIPPID_LLM_MODEL": ("model", str),
    "SHIPPID_LLM_API_KEY": ("api_key", str),
    "SHIPPID_LLM_TIMEOUT": ("timeout", float),
    "SHIPPID_LLM_CACHE_DIR": ("cache_dir", str),
    "SHIPPID_LLM_CONCURRENCY": ("concurrency", int),
    "SHIPPID_LLM_JSON_MODE": ("json_mode", str),
    "SHIPPID_LLM_CONTEXT_TOKENS": ("context_tokens", int),
}


def _apply(dc: Any, values: dict[str, Any]) -> None:
    names = {f.name for f in fields(dc)}
    for k, v in values.items():
        if k in names and v is not None:
            setattr(dc, k, v)


def load_config(path: Optional[str | Path] = None, **llm_overrides: Any) -> Config:
    cfg = Config()
    candidates = [path, os.environ.get("SHIPPID_CONFIG"), "shippid.yaml"]
    for cand in candidates:
        if cand and Path(cand).is_file():
            data = yaml.safe_load(Path(cand).read_text(encoding="utf-8")) or {}
            _apply(cfg.llm, data.get("llm", {}) or {})
            _apply(cfg, {k: v for k, v in data.items() if k != "llm"})
            break
    for env, (name, typ) in _ENV_MAP.items():
        if os.environ.get(env):
            setattr(cfg.llm, name, typ(os.environ[env]))
    _apply(cfg.llm, llm_overrides)
    return cfg
