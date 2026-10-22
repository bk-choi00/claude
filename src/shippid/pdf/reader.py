"""Low-level reading of a readable (vector + text) PDF page.

:func:`read_page` turns one PyMuPDF page into a :class:`PageContent`: straight
segments with their stroke style, circles recognised from Bezier (or dense
polygon) loops, filled polygons, and text words with font size and reading
direction. Everything is expressed in the *visible* page space (PDF points,
origin top-left, y down, page ``/Rotate`` applied), so coordinates match a
rendering of the page.

Only what is visible is read: vector content is clipped to the CropBox and to
the active clip paths (CAD viewports, cropped plots; approximated by the clip's
bounding rectangle), and text hidden by clip paths is dropped. Invisible text
(render mode 3 / zero alpha / neither filled nor stroked, e.g. an OCR layer or
AutoCAD ``PDFSHX=2`` text over SHX stroke glyphs) is kept but flagged, and does
not count as readable content. AutoCAD ``PDFSHX=1`` exports carry the strings of
SHX text (drawn as stroke geometry) in "AutoCAD SHX Text" annotations; these are
read as words flagged ``annotation``.

Curves tessellated into straight pieces (CAD exports of circles, arcs and hop
bridges) are recognised: closed regular n-gons (n >= 12) become circles, runs of
equal-turn pieces on a circular arc inside an open polyline are flagged as curve
pieces like flattened Beziers.

The reader does no P&ID interpretation; :mod:`shippid.extract` builds on it.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pymupdf

Color = Optional[tuple[float, float, float]]

_NO_DASH_RE = re.compile(r"^\[\s*\]")
try:  # stext character flags: a glyph that is neither filled nor stroked is invisible (render mode 3)
    from pymupdf import mupdf as _mupdf

    _STEXT_FILLED = int(getattr(_mupdf, "FZ_STEXT_FILLED", 16))
    _STEXT_STROKED = int(getattr(_mupdf, "FZ_STEXT_STROKED", 32))
except Exception:  # pragma: no cover - PyMuPDF builds without the low-level bindings
    _STEXT_FILLED, _STEXT_STROKED = 16, 32
_SHX_ANNOT_RE = re.compile(r"\bSHX\b", re.I)  # "AutoCAD SHX Text" (subject / author of PDFSHX=1 comments)
_DA_SIZE_RE = re.compile(r"([0-9]*\.?[0-9]+)\s+Tf\b")


@dataclass
class Segment:
    """A straight stroke piece. Curves are flattened into several segments."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float = 0.0
    color: Color = None
    dashed: bool = False
    path: int = -1  # index of the source path in ``page.get_drawings()``
    sub: int = 0  # subpath index within the path
    closed: bool = False  # part of a closed ring (polygon outline)
    filled: bool = False  # the source path is filled
    curve: int = -1  # >= 0: id of the Bezier this piece was flattened from
    kind: str = "l"  # source item: 'l' | 're' | 'qu' | 'c'
    layer: str = ""

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle(self) -> float:
        """Direction in degrees in [0, 180)."""
        return math.degrees(math.atan2(self.y1 - self.y0, self.x1 - self.x0)) % 180.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(self.x0, self.x1), min(self.y0, self.y1), max(self.x0, self.x1), max(self.y0, self.y1))


@dataclass
class CircleShape:
    cx: float
    cy: float
    r: float
    filled: bool = False
    stroked: bool = True
    width: float = 0.0
    color: Color = None
    dashed: bool = False
    path: int = -1
    sub: int = 0
    layer: str = ""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r)


@dataclass
class FilledPolygon:
    pts: list[tuple[float, float]]
    color: Color = None
    path: int = -1
    sub: int = 0
    layer: str = ""

    @property
    def area(self) -> float:
        a = 0.0
        n = len(self.pts)
        for i in range(n):
            x0, y0 = self.pts[i]
            x1, y1 = self.pts[(i + 1) % n]
            a += x0 * y1 - x1 * y0
        return abs(a) / 2.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class TextWord:
    """A whitespace-separated word (or a run of characters without large gaps)."""

    text: str
    bbox: tuple[float, float, float, float]
    size: float
    font: str = ""
    rotation: int = 0  # reading direction, degrees counter-clockwise (0 / 90 / 180 / 270 ...)
    color: Color = None
    line: int = 0  # index of the PDF text line the word came from
    origin: tuple[float, float] = (0.0, 0.0)
    hidden: bool = False  # invisible text (render mode 3 / zero alpha), e.g. an OCR layer or PDFSHX=2 text
    annotation: bool = False  # string of an "AutoCAD SHX Text" annotation (PDFSHX=1); bbox = annotation rect


@dataclass
class TextSpan:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    font: str = ""
    flags: int = 0
    rotation: int = 0
    color: Color = None
    line: int = 0
    origin: tuple[float, float] = (0.0, 0.0)
    hidden: bool = False
    annotation: bool = False


SCAN_MAX_PATHS = 40  # a page mostly covered by images with fewer visible vector paths is a scan


@dataclass
class PageContent:
    width: float
    height: float
    rotation: int = 0  # the page's /Rotate value
    segments: list[Segment] = field(default_factory=list)
    circles: list[CircleShape] = field(default_factory=list)
    polygons: list[FilledPolygon] = field(default_factory=list)  # filled, non-circular
    spans: list[TextSpan] = field(default_factory=list)
    words: list[TextWord] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)  # OCG names defined in the document
    image_coverage: float = 0.0  # fraction of the page area covered by raster images
    n_paths: int = 0  # visible vector paths (stroked or filled, not clipped away)
    n_chars: int = 0  # visible characters
    n_hidden_chars: int = 0  # invisible characters (OCR layers, PDFSHX=2 text)
    n_annot_chars: int = 0  # characters of "AutoCAD SHX Text" annotations (PDFSHX=1)
    n_clipped_paths: int = 0  # paths dropped entirely by the CropBox / clip paths
    n_arcs: int = 0  # tessellated arcs (runs of straight pieces) flagged as curves

    @property
    def empty(self) -> bool:
        """Nothing visible or searchable on the page."""
        return (self.n_paths == 0 and self.n_chars == 0 and self.n_hidden_chars == 0 and self.n_annot_chars == 0
                and self.image_coverage < 0.01)

    @property
    def scanned(self) -> bool:
        """Raster-dominated page (a scan, possibly with an invisible OCR text layer)."""
        return self.image_coverage > 0.6 and self.n_paths < SCAN_MAX_PATHS

    @property
    def readable(self) -> bool:
        """True when the page carries visible vector paths or real text and is not a scan."""
        return (self.n_paths > 0 or self.n_chars > 0) and not self.scanned


# --------------------------------------------------------------------------- helpers


def _rgb(c) -> Color:
    if c is None:
        return None
    if isinstance(c, (int, float)) and not isinstance(c, bool):  # sRGB int from get_text
        v = int(c)
        return ((v >> 16 & 255) / 255.0, (v >> 8 & 255) / 255.0, (v & 255) / 255.0)
    c = tuple(float(v) for v in c)
    if len(c) == 1:
        return (c[0], c[0], c[0])
    if len(c) == 4:  # CMYK
        cc, m, y, k = c
        return ((1 - cc) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return c[:3]  # type: ignore[return-value]


def _is_white(c: Color, thr: float = 0.97) -> bool:
    return c is not None and min(c) >= thr


def _bezier(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    mt = 1 - t
    a, b, c, d = mt * mt * mt, 3 * mt * mt * t, 3 * mt * t * t, t * t * t
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def _fit_circle(pts: list[tuple[float, float]]) -> Optional[tuple[float, float, float]]:
    """Centre/radius if all points lie on one circle (relative tolerance 5 %)."""
    if len(pts) < 6:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    ds = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    r = sum(ds) / len(ds)
    if r <= 1e-6:
        return None
    if max(abs(d - r) for d in ds) > 0.05 * r + 0.02:
        return None
    # the points must go all the way round (not a dense arc)
    angs = sorted(math.atan2(p[1] - cy, p[0] - cx) for p in pts)
    gaps = [b - a for a, b in zip(angs, angs[1:])] + [angs[0] + 2 * math.pi - angs[-1]]
    if max(gaps) > math.radians(100):
        return None
    return (cx, cy, r)


def _n_flatten(p0, p1, p2, p3) -> int:
    approx_len = math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
    return max(2, min(16, int(math.ceil(approx_len / 1.5))))


ARC_MIN_PIECES = 4  # a tessellated arc has at least this many straight pieces
ARC_MAX_TURN = 40.0  # degrees between consecutive pieces (a 9-gon or finer)
ARC_MIN_SWEEP = 40.0  # total turning of an arc run (degrees)


def _fit_arc(pts: list[tuple[float, float]]) -> Optional[tuple[float, float, float]]:
    """Least-squares circle (Kasa) through the points if they all lie on it (3 % of r)."""
    a = np.asarray(pts, dtype=float)
    A = np.column_stack((a[:, 0], a[:, 1], np.ones(len(a))))
    b = -(a[:, 0] ** 2 + a[:, 1] ** 2)
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return None
    cx, cy = -D / 2, -E / 2
    r2 = cx * cx + cy * cy - F
    if r2 <= 1e-12:
        return None
    r = math.sqrt(r2)
    d = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
    if float(np.max(np.abs(d - r))) > 0.03 * r + 0.02:
        return None
    return (float(cx), float(cy), r)


def _arc_runs(sp: list) -> list[tuple[int, int, float]]:
    """Runs ``[i, j)`` of consecutive straight items of a subpath that tessellate one circular
    arc (CAD exports flatten arcs, hop bridges and circles into polylines), with their sweep
    in degrees: at least :data:`ARC_MIN_PIECES` pieces of near-equal length turning by a
    near-equal angle in one sense, all vertices on a circle. Zigzags (alternating turns),
    sine waves (unequal turns) and polygons with few sides do not qualify."""
    n = len(sp)
    if n < ARC_MIN_PIECES:
        return []
    lens: list[float] = []
    angs: list[Optional[float]] = []
    for k, pts, _ in sp:
        if k != "l":
            lens.append(0.0)
            angs.append(None)
            continue
        (x0, y0), (x1, y1) = pts
        L = math.hypot(x1 - x0, y1 - y0)
        lens.append(L)
        angs.append(math.degrees(math.atan2(y1 - y0, x1 - x0)) if L > 1e-6 else None)

    def turn(i: int) -> Optional[float]:  # turning from piece i to piece i + 1
        a, b = angs[i], angs[i + 1]
        if a is None or b is None:
            return None
        return (b - a + 180.0) % 360.0 - 180.0

    runs: list[tuple[int, int, float]] = []
    i = 0
    while i <= n - ARC_MIN_PIECES:
        if angs[i] is None:
            i += 1
            continue
        lo = hi = lens[i]
        tmin = tmax = None
        sign = 0
        j = i + 1
        while j < n and angs[j] is not None:
            t = turn(j - 1)
            if t is None or not (1.0 <= abs(t) <= ARC_MAX_TURN):
                break
            s = 1 if t > 0 else -1
            if sign and s != sign:
                break
            nlo, nhi = min(lo, lens[j]), max(hi, lens[j])
            ntmin = abs(t) if tmin is None else min(tmin, abs(t))
            ntmax = abs(t) if tmax is None else max(tmax, abs(t))
            if nhi > 1.35 * nlo or ntmax > 1.6 * ntmin:
                break
            sign, lo, hi, tmin, tmax = s, nlo, nhi, ntmin, ntmax
            j += 1
        if j - i >= ARC_MIN_PIECES:
            sweep = sum(abs(turn(k)) for k in range(i, j - 1)) + (tmin + tmax) / 2  # + one piece's share
            pts = [sp[i][1][0]] + [sp[k][1][1] for k in range(i, j)]
            if sweep >= ARC_MIN_SWEEP and _fit_arc(pts) is not None:
                runs.append((i, j, sweep))
                i = j
                continue
        i += 1
    return runs


Box = tuple[float, float, float, float]


def _clip_seg(a: tuple[float, float], b: tuple[float, float], r: Box) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Liang-Barsky clipping of segment a-b to the rectangle ``r`` (None if outside)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - r[0]), (dx, r[2] - a[0]), (-dy, a[1] - r[1]), (dy, r[3] - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


def _box_and(a: Box, b: Box) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _inside(inner: Box, outer: Box, tol: float) -> bool:
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol


def _disjoint(a: Box, b: Box, tol: float) -> bool:
    return a[0] > b[2] + tol or a[2] < b[0] - tol or a[1] > b[3] + tol or a[3] < b[1] - tol


def _visible_paths(page: "pymupdf.Page") -> tuple[list[tuple[dict, Optional[Box]]], int]:
    """Drawings with the clip rectangle that applies to each (in unrotated page space;
    None = fully visible) and the number of paths that are clipped away entirely.

    Clip paths are approximated by their bounding rectangle ("scissor"); nested clips
    intersect. The base clip is the visible page (CropBox)."""
    vis = pymupdf.Rect(page.rect) * page.derotation_matrix
    vis.normalize()
    base: Box = (vis.x0, vis.y0, vis.x1, vis.y1)
    try:
        drawings = page.get_drawings(extended=True)
    except Exception:  # pragma: no cover - very old PyMuPDF without extended mode
        drawings = page.get_drawings()
    out: list[tuple[dict, Optional[Box]]] = []
    stack: list[tuple[int, Box]] = []  # (level, effective clip)
    dropped = 0
    for d in drawings:
        lvl = int(d.get("level", 0) or 0)
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        cur = stack[-1][1] if stack else base
        kind = d.get("type")
        if kind == "clip":
            sc = d.get("scissor")
            r = (sc.x0, sc.y0, sc.x1, sc.y1) if sc is not None else cur
            stack.append((lvl, _box_and(cur, r)))
            continue
        if kind == "group":
            stack.append((lvl, cur))
            continue
        if not d.get("items"):
            continue
        rr = d.get("rect")
        rb: Box = (rr.x0, rr.y0, rr.x1, rr.y1) if rr is not None else cur
        w = 0.5 * float(d.get("width") or 0.0) + 0.01
        if cur[0] >= cur[2] or cur[1] >= cur[3] or _disjoint(rb, cur, w):
            dropped += 1
            continue
        out.append((d, None if _inside(rb, cur, w + 0.05) else cur))
    return out, dropped


def _add_clipped(content: "PageContent", sp, clip: Box, width: float, color: Color, dashed: bool, pidx: int, sidx: int,
                 layer: str) -> None:
    """The visible pieces of a subpath crossing the clip boundary, as open segments (a
    cut outline is no longer closed, a cut circle no longer a circle)."""
    for k, pts, _ in sp:
        if k == "c":
            n = _n_flatten(*pts)
            pieces = [(_bezier(*pts, i / n), _bezier(*pts, (i + 1) / n)) for i in range(n)]
        else:
            pieces = [(pts[0], pts[1])]
        for a, b in pieces:
            c = _clip_seg(a, b, clip)
            if c is None or math.dist(*c) < 1e-6:
                continue
            (x0, y0), (x1, y1) = c
            content.segments.append(Segment(x0, y0, x1, y1, width, color, dashed, pidx, sidx, False, False, -1,
                                            "c" if k == "c" else "l", layer))


# --------------------------------------------------------------------------- main


def read_page(page: "pymupdf.Page") -> PageContent:
    """Read vector paths, circles, filled shapes and text of one page."""
    rot = int(page.rotation or 0) % 360
    mat = page.rotation_matrix if rot else None
    prect = page.rect

    def tp(p) -> tuple[float, float]:
        if mat is not None:
            p = pymupdf.Point(p) * mat
        return (float(p[0]), float(p[1]))

    content = PageContent(width=float(prect.width), height=float(prect.height), rotation=rot)
    try:
        content.layers = [v.get("name", "") for v in page.parent.get_ocgs().values()]
    except Exception:  # pragma: no cover - damaged OCG dictionaries
        content.layers = []

    curve_id = 0
    visible, content.n_clipped_paths = _visible_paths(page)
    for pidx, (d, clip_u) in enumerate(visible):
        stroke_col = _rgb(d.get("color"))
        fill_col = _rgb(d.get("fill"))
        dtype = d.get("type") or ""
        stroked = "s" in dtype and stroke_col is not None and not _is_white(stroke_col)
        if d.get("stroke_opacity") is not None and (d.get("stroke_opacity") or 0) <= 0.01:
            stroked = False
        filled = "f" in dtype and fill_col is not None and not _is_white(fill_col, 0.9)
        if d.get("fill_opacity") is not None and (d.get("fill_opacity") or 0) <= 0.01:
            filled = False
        if not stroked and not filled:
            continue
        content.n_paths += 1
        width = float(d.get("width") or 0.0)
        dashes = str(d.get("dashes") or "")
        dashed = bool(dashes) and not _NO_DASH_RE.match(dashes.strip()) and bool(re.search(r"[1-9]", dashes.split("]")[0]))
        layer = str(d.get("layer") or "")
        color = stroke_col if stroked else fill_col
        clip: Optional[Box] = None
        if clip_u is not None:  # partially visible path: clip rectangle in visible page space
            c0, c1 = tp((clip_u[0], clip_u[1])), tp((clip_u[2], clip_u[3]))
            clip = (min(c0[0], c1[0]), min(c0[1], c1[1]), max(c0[0], c1[0]), max(c0[1], c1[1]))

        # ---- split the path into subpaths: lists of (kind, pts, curve_id)
        subpaths: list[list[tuple[str, list[tuple[float, float]], int]]] = []
        cur: list[tuple[str, list[tuple[float, float]], int]] = []
        last = None
        for it in d["items"]:
            op = it[0]
            if op == "l":
                a, b = tp(it[1]), tp(it[2])
                if last is not None and math.dist(last, a) > 1e-3:
                    subpaths.append(cur)
                    cur = []
                cur.append(("l", [a, b], -1))
                last = b
            elif op == "c":
                pts = [tp(it[1]), tp(it[2]), tp(it[3]), tp(it[4])]
                if last is not None and math.dist(last, pts[0]) > 1e-3:
                    subpaths.append(cur)
                    cur = []
                cur.append(("c", pts, curve_id))
                curve_id += 1
                last = pts[3]
            elif op in ("re", "qu"):
                if cur:
                    subpaths.append(cur)
                    cur = []
                if op == "re":
                    r = pymupdf.Rect(it[1])
                    corners = [tp(r.tl), tp(r.tr), tp(r.br), tp(r.bl)]
                else:
                    q = it[1]
                    corners = [tp(q.ul), tp(q.ur), tp(q.lr), tp(q.ll)]
                ring = [(op, [corners[i], corners[(i + 1) % 4]], -1) for i in range(4)]
                subpaths.append(ring)
                last = None
        if cur:
            subpaths.append(cur)

        for sidx, sp in enumerate(subpaths):
            if not sp:
                continue
            if clip is not None:
                xs = [x for _, pts, _ in sp for x, _ in pts]
                ys = [y for _, pts, _ in sp for _, y in pts]
                sbox = (min(xs), min(ys), max(xs), max(ys))
                if _disjoint(sbox, clip, 0.5 * width):
                    continue
                if not _inside(sbox, clip, 0.5 * width + 0.05):
                    _add_clipped(content, sp, clip, width, color, dashed, pidx, sidx, layer)
                    continue
            start = sp[0][1][0]
            end = sp[-1][1][-1]
            closed = math.dist(start, end) <= 1e-2 or sp[0][0] in ("re", "qu")
            if not closed and d.get("closePath") and len(sp) >= 2:
                sp = sp + [("l", [end, start], -1)]
                closed = True
            if not closed and filled and len(sp) >= 2:
                closed = True  # a fill implicitly closes the subpath
                if math.dist(start, end) > 1e-2:
                    sp = sp + [("l", [end, start], -1)]

            # ---- tessellated arcs in open polylines: flag the pieces as one curve (like a flattened Bezier)
            if not closed and len(sp) >= ARC_MIN_PIECES:
                runs = _arc_runs(sp)
                if len(runs) == 1 and runs[0][:2] == (0, len(sp)) and runs[0][2] >= 345.0:
                    chord = sum(math.dist(*pts) for _, pts, _ in sp) / len(sp)
                    circ = _fit_arc([sp[0][1][0]] + [pts[1] for _, pts, _ in sp])
                    if circ is not None and math.dist(start, end) <= 2.0 * chord:  # an unclosed polygon circle
                        content.circles.append(
                            CircleShape(circ[0], circ[1], circ[2], filled=False, stroked=stroked, width=width,
                                        color=color, dashed=dashed, path=pidx, sub=sidx, layer=layer))
                        content.n_arcs += 1
                        continue
                if runs:
                    sp = list(sp)
                    for i0, j0, _ in runs:
                        for k in range(i0, j0):
                            sp[k] = ("l", sp[k][1], curve_id)
                        curve_id += 1
                    content.n_arcs += len(runs)

            # ---- circle recognition (Bezier loops or dense regular polygons)
            if closed:
                n_curves = sum(1 for k, _, _ in sp if k == "c")
                ring_pts: list[tuple[float, float]] = []
                for k, pts, _ in sp:
                    if k == "c":
                        ring_pts.extend(_bezier(*pts, t) for t in (0.0, 0.25, 0.5, 0.75))
                    else:
                        ring_pts.append(pts[0])
                circ = None
                if (n_curves >= 2 and n_curves == len(sp)) or (n_curves == 0 and len(sp) >= 12):
                    circ = _fit_circle(ring_pts)
                if circ is not None:
                    content.circles.append(
                        CircleShape(circ[0], circ[1], circ[2], filled=filled, stroked=stroked, width=width,
                                    color=color, dashed=dashed, path=pidx, sub=sidx, layer=layer)
                    )
                    continue
                if filled:
                    poly: list[tuple[float, float]] = []
                    for k, pts, _ in sp:
                        if k == "c":
                            n = _n_flatten(*pts)
                            poly.extend(_bezier(*pts, i / n) for i in range(n))
                        else:
                            poly.append(pts[0])
                    if len(poly) >= 3:
                        content.polygons.append(FilledPolygon(poly, color=fill_col, path=pidx, sub=sidx, layer=layer))

            # ---- segments (outline of strokes and of filled shapes)
            for k, pts, cid in sp:
                if k == "c":
                    n = _n_flatten(*pts)
                    prev = pts[0]
                    for i in range(1, n + 1):
                        q = _bezier(*pts, i / n)
                        content.segments.append(
                            Segment(prev[0], prev[1], q[0], q[1], width, color, dashed, pidx, sidx, closed, filled, cid, "c", layer)
                        )
                        prev = q
                else:
                    a, b = pts
                    if math.dist(a, b) < 1e-6:
                        continue
                    content.segments.append(Segment(a[0], a[1], b[0], b[1], width, color, dashed, pidx, sidx, closed, filled,
                                                    cid, "c" if cid >= 0 else k, layer))

    # ---- text
    # TEXT_CLIP (== TEXT_MEDIABOX_CLIP in older releases): drop glyphs outside the page and clip paths
    clip_flag = getattr(pymupdf, "TEXT_CLIP", 0) | getattr(pymupdf, "TEXT_MEDIABOX_CLIP", 0)
    td = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES | clip_flag)
    page_box = (-0.5, -0.5, content.width + 0.5, content.height + 0.5)
    line_no = 0
    n_chars = n_hidden = 0
    try:
        off_layers = {str(v.get("name", "")) for v in page.parent.get_ocgs().values() if not v.get("on", True)}
    except Exception:  # pragma: no cover - damaged OCG dictionaries
        off_layers = set()
    off_trace: Optional[list[tuple[str, Box]]] = None
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            if mat is not None:
                v = pymupdf.Point(dx, dy) * pymupdf.Matrix(mat.a, mat.b, mat.c, mat.d, 0, 0)
                dx, dy = v.x, v.y
            rotation = int(round(math.degrees(math.atan2(-dy, dx)))) % 360
            horizontal_read = abs(dx) >= abs(dy)
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                if not chars:
                    continue
                size = float(span.get("size", 0.0))
                font = str(span.get("font", ""))
                scol = _rgb(span.get("color"))
                text = "".join(c["c"] for c in chars)
                if not text.strip():
                    continue
                sb = _tbox(span["bbox"], tp)
                if _disjoint(sb, page_box, 0.0):
                    continue
                cflags = span.get("char_flags")
                hidden = span.get("alpha", 255) == 0 or (
                    cflags is not None and not int(cflags) & (_STEXT_FILLED | _STEXT_STROKED))
                if hidden and off_layers:
                    if off_trace is None:
                        off_trace = _off_layer_invisible_text(page, tp, off_layers)
                    if _in_off_trace(text, sb, off_trace):
                        continue  # invisible text on a layer that is switched off (MuPDF still extracts it)
                if hidden:
                    n_hidden += len(text.strip())
                else:
                    n_chars += len(text.strip())
                content.spans.append(TextSpan(text.strip(), sb, size, font, int(span.get("flags", 0)), rotation, scol,
                                              line_no, tp(span.get("origin", (sb[0], sb[3]))), hidden))
                # words: split at whitespace and at large gaps between glyphs
                word: list[tuple[str, tuple[float, float, float, float]]] = []
                for c in chars + [None]:
                    if c is not None:
                        cb = _tbox(c["bbox"], tp)
                    if c is None or c["c"].isspace() or (
                        word and _gap(word[-1][1], cb, horizontal_read) > 0.6 * max(size, 1.0)
                    ):
                        if word:
                            wt = "".join(w[0] for w in word)
                            x0 = min(w[1][0] for w in word)
                            y0 = min(w[1][1] for w in word)
                            x1 = max(w[1][2] for w in word)
                            y1 = max(w[1][3] for w in word)
                            if not _disjoint((x0, y0, x1, y1), page_box, 0.0):
                                content.words.append(TextWord(wt, (x0, y0, x1, y1), size, font, rotation, scol, line_no,
                                                              (x0, y1), hidden))
                        word = []
                        if c is None or c["c"].isspace():
                            continue
                    word.append((c["c"], cb))
            line_no += 1
    content.n_chars = n_chars
    content.n_hidden_chars = n_hidden
    for w in _shx_annotations(page, tp, mat, page_box, line_no):
        content.words.append(w)
        content.spans.append(TextSpan(w.text, w.bbox, w.size, w.font, 0, w.rotation, w.color, w.line, w.origin,
                                      annotation=True))
        content.n_annot_chars += len(w.text.replace(" ", ""))
    content.words = _dedupe_words(content.words)

    # ---- raster images
    try:
        infos = page.get_image_info()
    except Exception:  # pragma: no cover
        infos = []
    if infos:
        from shapely.geometry import box
        from shapely.ops import unary_union

        boxes = []
        for info in infos:
            b = _tbox(info["bbox"], tp)
            bx = box(max(0, b[0]), max(0, b[1]), min(content.width, b[2]), min(content.height, b[3]))
            if not bx.is_empty and bx.area > 0:
                boxes.append(bx)
        if boxes:
            content.image_coverage = float(unary_union(boxes).area / max(1.0, content.width * content.height))
    return content


def _off_layer_invisible_text(page: "pymupdf.Page", tp, off_layers: set[str]) -> list[tuple[str, Box]]:
    """(text, bbox) of invisible text spans inside optional content that is OFF. MuPDF drops
    visible OFF-layer text from ``get_text`` but still reports invisible (render mode 3)
    text there; the text trace knows each span's layer."""
    out: list[tuple[str, Box]] = []
    try:
        trace = page.get_texttrace()
    except Exception:  # pragma: no cover - very old PyMuPDF
        return out
    for t in trace:
        if t.get("layer") in off_layers:
            text = "".join(chr(c[0]) for c in t.get("chars", ()) if c[0] > 0)
            out.append((re.sub(r"\s+", "", text), _tbox(t["bbox"], tp)))
    return out


def _in_off_trace(text: str, bbox: Box, trace: list[tuple[str, Box]]) -> bool:
    t = re.sub(r"\s+", "", text)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    pad = 0.5 * max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
    for s, b in trace:
        if t and t in s and b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad:
            return True
    return False


def _shx_annotations(page: "pymupdf.Page", tp, mat, page_box, line0: int) -> list[TextWord]:
    """Strings of AutoCAD ``PDFSHX=1`` exports: SHX text is drawn as stroke geometry and each
    text entity is stored as an annotation (subject / author "AutoCAD SHX Text") whose
    contents are the string and whose rectangle covers it. One :class:`TextWord` per line of
    the contents (``annotation=True``), in displayed page coordinates. The rectangle is read
    from the annotation dictionary (MuPDF reports a fixed icon box for 'Text' notes); the
    reading direction is the FreeText ``/Rotate`` (else: vertical when the box is tall)."""
    out: list[TextWord] = []
    try:
        annots = list(page.annots())
    except Exception:  # pragma: no cover - damaged annotation arrays
        return out
    doc = page.parent
    try:
        layer_on = {int(x): bool(v.get("on", True)) for x, v in doc.get_ocgs().items()}
    except Exception:  # pragma: no cover - damaged OCG dictionaries
        layer_on = {}
    for a in annots:
        try:
            info = a.info or {}
            tag = " ".join(str(info.get(k) or "") for k in ("subject", "title", "name"))
            if not _SHX_ANNOT_RE.search(tag):
                continue
            oc = a.get_oc()
            if oc and not layer_on.get(int(oc), True):
                continue  # on a layer that is switched off, like the strokes it describes
            content = str(info.get("content") or "").replace("\r", "\n").strip()
            if not content:
                continue
            rect = a.rect
            kind, val = doc.xref_get_key(a.xref, "Rect")
            if kind == "array":
                nums = [float(v) for v in re.findall(r"[-+]?[0-9]*\.?[0-9]+", val)]
                if len(nums) == 4:
                    rect = pymupdf.Rect(nums) * page.transformation_matrix
            rect.normalize()
            size = None
            dk, dv = doc.xref_get_key(a.xref, "DA")
            if dk == "string":
                m = _DA_SIZE_RE.search(dv)
                if m and float(m.group(1)) > 0:
                    size = float(m.group(1))
            rot_u = int(a.rotation) if a.type[1] == "FreeText" and (a.rotation or 0) > 0 else None
        except Exception:  # pragma: no cover - malformed annotation
            continue
        bb = _tbox((rect.x0, rect.y0, rect.x1, rect.y1), tp)
        if bb[2] - bb[0] <= 0 or bb[3] - bb[1] <= 0 or _disjoint(bb, page_box, 0.0):
            continue
        texts = [t.strip() for t in content.split("\n") if t.strip()]
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if rot_u is not None:
            dx, dy = math.cos(math.radians(rot_u)), -math.sin(math.radians(rot_u))
            if mat is not None:
                v = pymupdf.Point(dx, dy) * pymupdf.Matrix(mat.a, mat.b, mat.c, mat.d, 0, 0)
                dx, dy = v.x, v.y
            rotation = int(round(math.degrees(math.atan2(-dy, dx)))) % 360
        else:
            longest = max(len(t) for t in texts)
            rotation = 90 if h > 1.5 * w and (longest >= 2 or len(texts) >= 2) else 0
        vertical = rotation % 180 == 90
        across = (w if vertical else h) / len(texts)  # line pitch across the reading direction
        if size is None:
            size = across / 1.3
        size = min(size, across)
        for k, t in enumerate(texts):
            if vertical:  # successive lines run against the glyphs' up direction: rightwards for 90
                x0 = bb[0] + k * across if rotation == 90 else bb[2] - (k + 1) * across
                box = (x0, bb[1], x0 + across, bb[3])
                origin = (box[2] if rotation == 90 else box[0], box[3] if rotation == 90 else box[1])
            else:
                box = (bb[0], bb[1] + k * across, bb[2], bb[1] + (k + 1) * across)
                origin = (box[0], box[3])
            out.append(TextWord(re.sub(r"\s+", " ", t), box, size, "SHX", rotation, None, line0 + len(out), origin,
                                hidden=False, annotation=True))
    return out


def _dedupe_words(words: list[TextWord]) -> list[TextWord]:
    """Drop a word drawn twice at the same place (duplicated CAD text entities)."""
    seen: set[tuple] = set()
    out = []
    for w in words:
        key = (w.text, w.rotation, w.hidden, w.annotation) + tuple(int(round(v * 10)) for v in w.bbox)
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def _tbox(b, tp) -> tuple[float, float, float, float]:
    p0 = tp((b[0], b[1]))
    p1 = tp((b[2], b[3]))
    return (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))


def _gap(a, b, horizontal: bool) -> float:
    if horizontal:
        return max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))


log = logging.getLogger(__name__)

MAX_WARNINGS = 20  # MuPDF messages kept in the document meta


def open_pdf(path) -> "pymupdf.Document":
    """Open a PDF with PyMuPDF (raises ``ValueError`` for encrypted documents). MuPDF's warning store
    is cleared first, so :func:`pdf_health` reports what this document produces; a file MuPDF had to
    repair (broken cross-reference table, truncated download) is logged."""
    pymupdf.TOOLS.mupdf_warnings(reset=True)
    doc = pymupdf.open(str(path))
    if doc.needs_pass:
        doc.close()
        raise ValueError(f"{path}: encrypted PDF")
    if getattr(doc, "is_repaired", False):
        msgs = [m for m in pymupdf.TOOLS.mupdf_warnings(reset=False).splitlines() if m.strip()]
        log.warning("%s: damaged PDF, repaired by MuPDF (%s); content may be missing", path,
                    "; ".join(msgs[:3]) or "no details")
    return doc


def pdf_health(doc: "pymupdf.Document") -> dict[str, Any]:
    """Damage indicators of an open document: ``{"repaired": bool, "warnings": [...], "n_warnings": n}``
    -- MuPDF repaired the file on opening, and the warnings / errors it reported since :func:`open_pdf`
    (corrupt or truncated content streams, missing objects, broken fonts), which mean the drawing as
    read may be incomplete. Empty for an intact file. Reading the warnings clears MuPDF's store."""
    msgs = [m.strip() for m in pymupdf.TOOLS.mupdf_warnings(reset=True).splitlines() if m.strip()]
    repaired = bool(getattr(doc, "is_repaired", False))
    if not repaired and not msgs:
        return {}
    uniq = list(dict.fromkeys(m for m in msgs if not m.startswith("... repeated")))
    return {"repaired": repaired, "warnings": uniq[:MAX_WARNINGS], "n_warnings": len(msgs)}
