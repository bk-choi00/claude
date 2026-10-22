"""Text metrics for base-14 fonts, matching what PyMuPDF text extraction reports.

A span inserted at baseline origin ``(x, y)`` with font size ``fs`` occupies
``[x, x + width] x [y - asc*fs, y - desc*fs]`` (``desc`` is negative). Text
rotated by 90 degrees (reading bottom to top, as used for line numbers on
vertical pipes) occupies ``[x - asc*fs, x - desc*fs] x [y - width, y]``.
"""

from __future__ import annotations

from functools import lru_cache

import pymupdf

# PyMuPDF short names of the base-14 fonts used by the generator
REGULAR_FONTS = ("helv", "tiro", "cour")
BOLD_FONTS = {"helv": "hebo", "tiro": "tibo", "cour": "cobo", "hebo": "hebo", "tibo": "tibo", "cobo": "cobo"}


@lru_cache(maxsize=None)
def _font(name: str) -> pymupdf.Font:
    return pymupdf.Font(name)


@lru_cache(maxsize=None)
def font_metrics(name: str) -> tuple[float, float]:
    """(ascender, descender) as fractions of the font size."""
    f = _font(name)
    return float(f.ascender), float(f.descender)


@lru_cache(maxsize=65536)
def text_width(text: str, font: str, size: float) -> float:
    return float(pymupdf.get_text_length(text, fontname=font, fontsize=size))


def line_height(font: str, size: float) -> float:
    asc, desc = font_metrics(font)
    return (asc - desc) * size


def text_bbox(text: str, x: float, y: float, size: float, font: str, rotate: int = 0) -> tuple[float, float, float, float]:
    """Bounding box of a single-line text drawn with its baseline origin at ``(x, y)``."""
    w = text_width(text, font, size)
    asc, desc = font_metrics(font)
    if rotate == 0:
        return (x, y - asc * size, x + w, y - desc * size)
    if rotate == 90:
        return (x - asc * size, y - w, x - desc * size, y)
    raise ValueError("only 0 and 90 degree text is generated")


def origin_for_box(text: str, x0: float, y0: float, size: float, font: str, rotate: int = 0) -> tuple[float, float]:
    """Baseline origin that puts the text bbox's top-left corner at ``(x0, y0)``."""
    asc, desc = font_metrics(font)
    if rotate == 0:
        return (x0, y0 + asc * size)
    w = text_width(text, font, size)
    return (x0 + asc * size, y0 + w)


def wrap(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Greedy word wrap to ``max_width`` points."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if cur and text_width(cand, font, size) > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines
