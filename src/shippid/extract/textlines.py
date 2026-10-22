"""Assembly of PDF words into text lines (the unit that becomes a ``TextItem``).

A CAD text entity (one tag, one line number, one label line) must come out as one
line however the exporter wrote it:

* **fragments** -- per-character or per-word text objects with explicit positions
  are merged into the logical token: same reading direction, font and font size,
  aligned baselines, consecutive along the reading direction with a gap of at most
  :data:`FRAGMENT_GAP` em (a word space, also of monospaced fonts; a full-width space
  between CJK characters), and no table rule between them. Texts further apart or in
  another size stay separate (two tags side by side, a tag right after a label,
  neighbouring table cells).
* **invisible text** (render mode 3 / zero alpha) that duplicates visible text at the
  same place is dropped (it only groups the visible fragments it covers). Where no
  visible text exists it *is* the text layer: AutoCAD ``PDFSHX=2`` exports draw SHX
  text as stroke geometry and put an invisible copy of the string on top.
* **"AutoCAD SHX Text" annotations** (``PDFSHX=1``) are the text where neither visible
  nor invisible text exists.

Lines of the last two kinds have ``source`` ``"hidden"`` / ``"annotation"``: their
visible glyphs, if any, are vector strokes (see :attr:`TextLine.geometric`).
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from ..pdf.reader import TextWord

FRAGMENT_GAP = 0.66  # em: largest gap between two fragments of one text (a Courier space is 0.6 em)
CJK_GAP = 1.1  # em: between CJK characters (full-width spaces)
SPACE_GAP = 0.12  # em: a larger gap between two merged fragments stands for a space
SIZE_TOL = 0.02  # fragments of one text have the same font size (relative tolerance)
_CJK_RE = re.compile(r"[ᄀ-ᇿ　-ヿ㄰-㆏㐀-鿿가-힯豈-﫿＀-￯]")


@dataclass
class TextLine:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    rotation: int
    words: list[TextWord] = field(default_factory=list)
    font: str = ""
    order: int = 0  # appearance order on the page
    source: str = "text"  # 'text' | 'hidden' (invisible text is the text layer) | 'annotation' (SHX annotation)

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def vertical(self) -> bool:
        return self.rotation % 180 == 90

    @property
    def geometric(self) -> bool:
        """The visible glyphs of this text (if any) are vector strokes, not font glyphs."""
        return self.source != "text"


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


def _uv(bbox: Sequence[float], rotation: int) -> tuple[float, float, float, float]:
    """(u0, u1, v0, v1): extent along the reading direction (u grows in reading order) and
    across it, for the four axis-aligned directions."""
    x0, y0, x1, y1 = bbox
    r = rotation % 360
    if r == 90:
        return (-y1, -y0, x0, x1)
    if r == 180:
        return (-x1, -x0, -y1, -y0)
    if r == 270:
        return (y0, y1, -x1, -x0)
    return (x0, x1, y0, y1)


@dataclass
class _Piece:
    words: list[TextWord]
    first: int  # index of the first word in the input (appearance order)
    rotation: int
    size: float
    bbox: tuple[float, float, float, float]

    @property
    def uv(self) -> tuple[float, float, float, float]:
        return _uv(self.bbox, self.rotation)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def _box(words: Sequence[TextWord]) -> tuple[float, float, float, float]:
    return (min(w.bbox[0] for w in words), min(w.bbox[1] for w in words), max(w.bbox[2] for w in words),
            max(w.bbox[3] for w in words))


def _piece(words: list[TextWord], first: int) -> _Piece:
    return _Piece(words, first, words[0].rotation, max(w.size for w in words), _box(words))


def _join(a: TextWord, b: TextWord) -> TextWord:
    """Two fragments of one word (no space between them)."""
    return replace(a, text=a.text + b.text, bbox=_box([a, b]), size=max(a.size, b.size))


def _concat(a: _Piece, b: _Piece, gap: float) -> _Piece:
    size = max(a.size, b.size, 1e-6)
    if gap > SPACE_GAP * size:
        words = a.words + b.words
    else:
        words = a.words[:-1] + [_join(a.words[-1], b.words[0])] + b.words[1:]
    return _Piece(words, min(a.first, b.first), a.rotation, max(a.size, b.size), _box(words))


def _pdf_line_pieces(words: Sequence[TextWord], index: dict[int, int], gap_factor: float,
                     rules: "_Rules") -> list[_Piece]:
    """Words of the same PDF text line, split at large gaps, at a change of font size and at
    table rules (two separate texts that happen to share a baseline stay separate)."""
    by_line: dict[int, list[TextWord]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w)
    out: list[_Piece] = []
    for line_no in sorted(by_line):
        ws = sorted(by_line[line_no], key=_reading_key)
        groups: list[list[TextWord]] = [[ws[0]]]
        for w in ws[1:]:
            prev = groups[-1][-1]
            size = max(prev.size, w.size, 1.0)
            gap = _gap(prev, w)
            if gap > gap_factor * size or w.rotation != prev.rotation or \
                    (abs(prev.size - w.size) > SIZE_TOL * size and gap > SPACE_GAP * size) or \
                    rules.splits(prev.bbox, w.bbox, w.rotation, size):
                groups.append([w])
            else:
                groups[-1].append(w)
        for g in groups:
            out.append(_piece(g, min(index[id(w)] for w in g)))
    return out


class _Rules:
    """Strokes that separate texts: table rules / cell borders across the reading direction."""

    def __init__(self, segments: Optional[Sequence[tuple[float, float, float, float]]]):
        arr = np.asarray(segments if segments is not None else [], dtype=float).reshape(-1, 4)
        dx, dy = np.abs(arr[:, 2] - arr[:, 0]), np.abs(arr[:, 3] - arr[:, 1])
        self.v = arr[dx <= 0.02 * np.maximum(dy, 1e-9)]  # vertical strokes separate horizontal text
        self.h = arr[dy <= 0.02 * np.maximum(dx, 1e-9)]

    def between(self, a: _Piece, b: _Piece) -> bool:
        """A rule crosses the gap between ``a`` and ``b`` over most of the text height."""
        return self.splits(a.bbox, b.bbox, a.rotation, max(a.size, b.size))

    def splits(self, a: Sequence[float], b: Sequence[float], rotation: int, size: float) -> bool:
        vertical_text = rotation % 180 == 90
        segs = self.h if vertical_text else self.v
        if not len(segs):
            return False
        if vertical_text:  # gap along y, text height along x
            lo, hi = min(a[3], b[3]), max(a[1], b[1])
            c0, c1 = max(a[0], b[0]), min(a[2], b[2])
            pos = (segs[:, 1] + segs[:, 3]) / 2
            s0, s1 = np.minimum(segs[:, 0], segs[:, 2]), np.maximum(segs[:, 0], segs[:, 2])
        else:  # gap along x, text height along y
            lo, hi = min(a[2], b[2]), max(a[0], b[0])
            c0, c1 = max(a[1], b[1]), min(a[3], b[3])
            pos = (segs[:, 0] + segs[:, 2]) / 2
            s0, s1 = np.minimum(segs[:, 1], segs[:, 3]), np.maximum(segs[:, 1], segs[:, 3])
        if hi < lo:
            lo, hi = hi, lo
        span = max(c1 - c0, 1e-6)
        pad = 0.05 * size
        cover = np.minimum(s1, c1) - np.maximum(s0, c0)
        return bool(np.any((pos >= lo - pad) & (pos <= hi + pad) & (cover >= 0.7 * span)))


def _cjk_pair(a: _Piece, b: _Piece) -> bool:
    """Both fragments are CJK text (spaced with full-width spaces, also around brackets)."""
    return bool(_CJK_RE.search(a.text) and _CJK_RE.search(b.text))


def _link_gap(a: _Piece, b: _Piece) -> Optional[float]:
    """Gap (points) from ``a`` to ``b`` along the reading direction when ``b`` may continue
    ``a``, else None. Fragments of one text entity share its reading direction and font size
    exactly and sit on one baseline; a neighbouring entity in another size (a tag written
    right after an equipment label) is never joined, one in another font only when the two
    touch without a space (font fallback inside a word) or both are CJK."""
    if a.rotation != b.rotation or a.rotation % 90:
        return None
    size = max(a.size, b.size, 1e-6)
    if abs(a.size - b.size) > SIZE_TOL * size:
        return None
    au0, au1, av0, av1 = a.uv
    bu0, bu1, bv0, bv1 = b.uv
    if abs((av0 + av1) - (bv0 + bv1)) / 2 > 0.3 * size:
        return None
    if min(av1, bv1) - max(av0, bv0) < 0.6 * min(av1 - av0, bv1 - bv0):
        return None
    gap = bu0 - au1
    cjk = _cjk_pair(a, b)
    limit = (CJK_GAP if cjk else FRAGMENT_GAP) * size
    if a.words[-1].font != b.words[0].font and not cjk:
        limit = SPACE_GAP * size
    if gap < -0.25 * size or gap > limit or bu1 <= au1:
        return None
    return gap


def _merge_fragments(pieces: list[_Piece], rules: _Rules) -> list[_Piece]:
    """Chain pieces that continue each other (each piece links to its nearest successor)."""
    if len(pieces) < 2:
        return pieces
    order = sorted(range(len(pieces)), key=lambda k: (pieces[k].uv[2] + pieces[k].uv[3]) / 2)
    keys = [(pieces[k].uv[2] + pieces[k].uv[3]) / 2 for k in order]
    succ: dict[int, tuple[float, int]] = {}
    for k in range(len(pieces)):
        a = pieces[k]
        vc = (a.uv[2] + a.uv[3]) / 2
        lo = bisect.bisect_left(keys, vc - 0.3 * a.size * 1.3)
        hi = bisect.bisect_right(keys, vc + 0.3 * a.size * 1.3)
        best = None
        for idx in range(lo, hi):
            j = order[idx]
            if j == k:
                continue
            g = _link_gap(a, pieces[j])
            if g is not None and (best is None or g < best[0]):
                best = (g, j)
        if best is not None:
            succ[k] = best
    # resolve: a piece is continued by one piece and continues one piece (the nearest wins); a table
    # rule between two pieces cuts the link. Links always advance along the reading direction: no cycles.
    pred: dict[int, tuple[float, int]] = {}
    for k, (g, j) in succ.items():
        if (j not in pred or g < pred[j][0]) and not rules.between(pieces[k], pieces[j]):
            pred[j] = (g, k)
    nxt = {k: j for j, (_, k) in pred.items()}
    out: list[_Piece] = []
    for k in range(len(pieces)):
        if k in pred:
            continue  # continues another piece
        cur, j = pieces[k], nxt.get(k)
        while j is not None:
            cur = _concat(cur, pieces[j], pred[j][0])
            j = nxt.get(j)
        out.append(cur)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


class _Grid:
    """Bucket index of boxes (pages hold thousands of text pieces)."""

    CELL = 40.0

    def __init__(self, boxes: Sequence[Sequence[float]]):
        self.cells: dict[tuple[int, int], list[int]] = {}
        for k, b in enumerate(boxes):
            for key in self._keys(b, 0.0):
                self.cells.setdefault(key, []).append(k)

    def _keys(self, b: Sequence[float], pad: float):
        c = self.CELL
        for i in range(int(math.floor((b[0] - pad) / c)), int(math.floor((b[2] + pad) / c)) + 1):
            for j in range(int(math.floor((b[1] - pad) / c)), int(math.floor((b[3] + pad) / c)) + 1):
                yield (i, j)

    def query(self, b: Sequence[float], pad: float = 0.0) -> list[int]:
        out: set[int] = set()
        for key in self._keys(b, pad):
            out.update(self.cells.get(key, ()))
        return sorted(out)


def _group_under(hidden: list[_Piece], visible: list[_Piece]) -> list[_Piece]:
    """Visible fragments lying under one invisible copy of the same string (PDFSHX=2 style
    duplicates of fragmented text) form one text, even where their gaps are wide."""
    if not hidden or not visible:
        return visible
    taken: set[int] = set()
    merged: list[_Piece] = []
    grid = _Grid([v.bbox for v in visible])
    for h in hidden:
        hu0, hu1, hv0, hv1 = h.uv
        pad = 0.3 * h.size
        under = []
        for k in grid.query(h.bbox, pad):
            v = visible[k]
            if k in taken or v.rotation != h.rotation:
                continue
            vu0, vu1, vv0, vv1 = v.uv
            cu, cv = (vu0 + vu1) / 2, (vv0 + vv1) / 2
            if hu0 - pad <= cu <= hu1 + pad and hv0 - pad <= cv <= hv1 + pad:
                under.append(k)
        if len(under) < 2:
            continue
        under.sort(key=lambda k: visible[k].uv[0])
        if _norm("".join(visible[k].text for k in under)) != _norm(h.text):
            continue
        cur = visible[under[0]]
        for k in under[1:]:
            cur = _concat(cur, visible[k], visible[k].uv[0] - cur.uv[1])
        merged.append(cur)
        taken.update(under)
    return [v for k, v in enumerate(visible) if k not in taken] + merged


def _duplicate(p: _Piece, others: Sequence[_Piece], grid: Optional[_Grid] = None) -> bool:
    """``p`` (invisible text / annotation) repeats text that is already there."""
    area = max(_area(p.bbox), 1e-9)
    covered = 0.0
    target = _norm(p.text)
    for k in (grid.query(p.bbox) if grid is not None else range(len(others))):
        o = others[k]
        ov = _inter(p.bbox, o.bbox)
        if ov <= 0:
            continue
        if _norm(o.text) == target:
            return True
        covered += ov
    return covered >= 0.4 * area


def build_lines(words: Sequence[TextWord], gap_factor: float = 1.2,
                separators: Optional[Sequence[tuple[float, float, float, float]]] = None) -> list[TextLine]:
    """Logical text lines of a page (see the module docstring). ``separators``: straight
    strokes ``(x0, y0, x1, y1)`` of the page (table rules keep cell texts apart)."""
    index = {id(w): k for k, w in enumerate(words)}
    rules = _Rules(separators)
    vis = [w for w in words if not w.hidden and not w.annotation]
    hid = [w for w in words if w.hidden and not w.annotation]
    ann = [w for w in words if w.annotation]
    visible = _merge_fragments(_pdf_line_pieces(vis, index, gap_factor, rules), rules) if vis else []
    hidden = _merge_fragments(_pdf_line_pieces(hid, index, gap_factor, rules), rules) if hid else []
    visible = _group_under(hidden, visible)
    lines: list[tuple[_Piece, str]] = [(p, "text") for p in visible]
    vgrid = _Grid([v.bbox for v in visible])
    kept_hidden = [h for h in hidden if not _duplicate(h, visible, vgrid)]
    lines += [(h, "hidden") for h in kept_hidden]
    taken = visible + kept_hidden
    tgrid = _Grid([t.bbox for t in taken])
    for w in ann:
        p = _piece([w], index[id(w)])
        if not _duplicate(p, taken, tgrid):
            lines.append((p, "annotation"))
            taken.append(p)
            for key in tgrid._keys(p.bbox, 0.0):
                tgrid.cells.setdefault(key, []).append(len(taken) - 1)
    lines.sort(key=lambda ps: ps[0].first)
    out: list[TextLine] = []
    for p, source in lines:
        out.append(TextLine(p.text, p.bbox, p.size, p.rotation, list(p.words), p.words[0].font, len(out), source))
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
