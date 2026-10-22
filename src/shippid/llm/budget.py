"""Token budgeting for Gemma 4 31B requests (no tokenizer is installed: conservative estimates).

The documented vLLM recipe serves Gemma 4 31B with ``--max-model-len 32768``; a request whose
prompt plus ``max_tokens`` exceeds that is rejected with HTTP 400. Every step that sends a large
digest (the engineering review, the 3D planner) sizes its prompt and ``max_tokens`` with these
helpers against ``context_tokens`` (``LLMConfig.context_tokens``, carried by the client).
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

#: ``--max-model-len`` of the documented vLLM recipe (docs/RESEARCH.md)
DEFAULT_CONTEXT_TOKENS = 32768
#: per-message chat-template overhead (turn markers, role)
MESSAGE_OVERHEAD = 8
#: what LLMClient's repair round adds to a prompt: up to 6000 characters of the failed answer
#: plus up to 2000 characters of error text and the instruction around it
REPAIR_CHARS = 8200


def estimate_tokens(text: str) -> int:
    """Conservative token count of a prompt for Gemma's SentencePiece tokenizer.

    Digits are split into single tokens and non-ASCII characters (Korean titles) cost about one
    token each; the remaining JSON / English text is counted at 3 characters per token (real
    English prose is ~4, so this over-estimates on purpose).
    """
    digits = sum(1 for c in text if c.isdigit())
    wide = sum(1 for c in text if ord(c) > 127)
    return digits + wide + math.ceil((len(text) - digits - wide) / 3.0)


def schema_note_tokens(schema: dict[str, Any]) -> int:
    """Tokens of the schema-in-prompt note LLMClient appends to the last user turn."""
    return estimate_tokens(json.dumps(schema, ensure_ascii=False)) + 60


def messages_tokens(messages: Sequence[Any], schema: Optional[dict[str, Any]] = None) -> int:
    """Estimated prompt tokens of chat messages (text parts only; images are counted by the caller)."""
    total = sum(estimate_tokens(m.text()) + MESSAGE_OVERHEAD for m in messages)
    return total + (schema_note_tokens(schema) if schema is not None else 0)


def repair_reserve(max_attempts: int) -> int:
    """Context kept free for one repair round (failed answer + error text), 256 tokens otherwise."""
    return 256 + (estimate_tokens("x" * REPAIR_CHARS) if max_attempts > 1 else 0)


def context_of(client: Any, default: Optional[int] = DEFAULT_CONTEXT_TOKENS) -> Optional[int]:
    """The server context (tokens) a client was configured with (``LLMClient.context_tokens``)."""
    value = getattr(client, "context_tokens", None)
    return int(value) if isinstance(value, (int, float)) and value > 0 else default
