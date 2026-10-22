"""Robust JSON extraction from model output (thinking blocks, code fences, prose)."""

from __future__ import annotations

import json
import re
from typing import Any

# Thinking / reasoning delimiters used by open-weight chat templates. Servers
# with a reasoning parser strip these already; this is the fallback.
_THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.S | re.I),
    re.compile(r"<thinking>.*?</thinking>", re.S | re.I),
    re.compile(r"<\|channel\|?>thought.*?<channel\|>", re.S),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.S),
]
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def strip_thinking(text: str) -> str:
    for pat in _THINK_PATTERNS:
        text = pat.sub("", text)
    return text


def _match_end(text: str, start: int) -> int:
    """Index of the bracket closing ``text[start]`` (string-aware), or -1."""
    close = "}" if text[start] == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i if c == close else -1
    return -1


def _balanced_candidates(text: str):
    """Yield top-level {...} / [...] substrings using bracket matching that respects strings."""
    i = 0
    while i < len(text):
        if text[i] in "{[":
            end = _match_end(text, i)
            if end > 0:
                yield text[i : end + 1]
                i = end + 1
                continue
        i += 1


def _loads_lenient(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # trailing commas are the most common defect
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(fixed)


def extract_json(text: str) -> Any:
    """Return the first JSON value found in ``text``; raises ``ValueError`` if none."""
    text = strip_thinking(text).strip()
    if not text:
        raise ValueError("empty response")
    try:
        return _loads_lenient(text)
    except json.JSONDecodeError:
        pass
    for m in _FENCE.finditer(text):
        try:
            return _loads_lenient(m.group(1).strip())
        except json.JSONDecodeError:
            continue
    best = None
    for cand in _balanced_candidates(text):
        try:
            val = _loads_lenient(cand)
        except json.JSONDecodeError:
            continue
        if best is None or len(cand) > len(best[0]):
            best = (cand, val)
    if best is not None:
        return best[1]
    raise ValueError("no JSON object found in model output: " + text[:200])
