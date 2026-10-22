"""Assembly of PDF words into text lines (the unit that becomes a ``TextItem``)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..pdf.reader import TextWord


@dataclass
class TextLine:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    rotation: int
    words: list[TextWord] = field(default_factory=list)
    font: str = ""
    order: int = 0  # appearance order on the page

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def vertical(self) -> bool:
        return self.rotation % 180 == 90


def _reading_key(w: TextWord) -> float:
    r = w.rotation % 360
    if r == 90:
        return -w.bbox[3]
    if r == 270:
        return w.bbox[1]
    if r == 180:
        return -w.bbox[2]
    return w.bbox[0]


def _gap(a: TextWord, b: TextWord) -> float:
    if a.rotation % 180 == 90:
        return max(0.0, max(a.bbox[1], b.bbox[1]) - min(a.bbox[3], b.bbox[3]))
    return max(0.0, max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2]))


def build_lines(words: Sequence[TextWord], gap_factor: float = 1.2) -> list[TextLine]:
    """Group words of the same PDF line into lines, splitting at large gaps
    (two separate texts that happen to share a baseline stay separate)."""
    by_line: dict[int, list[TextWord]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w)
    out: list[TextLine] = []
    for line_no in sorted(by_line):
        ws = sorted(by_line[line_no], key=_reading_key)
        groups: list[list[TextWord]] = [[ws[0]]]
        for w in ws[1:]:
            prev = groups[-1][-1]
            size = max(prev.size, w.size, 1.0)
            if _gap(prev, w) > gap_factor * size or w.rotation != prev.rotation:
                groups.append([w])
            else:
                groups[-1].append(w)
        for g in groups:
            x0 = min(w.bbox[0] for w in g)
            y0 = min(w.bbox[1] for w in g)
            x1 = max(w.bbox[2] for w in g)
            y1 = max(w.bbox[3] for w in g)
            size = max(w.size for w in g)
            out.append(TextLine(" ".join(w.text for w in g), (x0, y0, x1, y1), size, g[0].rotation, list(g), g[0].font,
                                len(out)))
    return out


def line_height(t: TextLine) -> float:
    return (t.bbox[2] - t.bbox[0]) if t.vertical else (t.bbox[3] - t.bbox[1])


def median(vals: Sequence[float], default: float = 0.0) -> float:
    vals = sorted(vals)
    if not vals:
        return default
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def text_distance(t: TextLine, bbox: Sequence[float]) -> float:
    dx = max(0.0, max(t.bbox[0], bbox[0]) - min(t.bbox[2], bbox[2]))
    dy = max(0.0, max(t.bbox[1], bbox[1]) - min(t.bbox[3], bbox[3]))
    return math.hypot(dx, dy)
