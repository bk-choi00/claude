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
(render mode 3, e.g. an OCR layer) is kept but flagged, and does not count as
readable content.

The reader does no P&ID interpretation; :mod:`shippid.extract` builds on it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

Color = Optional[tuple[float, float, float]]

_NO_DASH_RE = re.compile(r"^\[\s*\]")


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
    hidden: bool = False  # invisible text (render mode 3 / zero alpha), e.g. an OCR layer


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
    n_hidden_chars: int = 0  # invisible characters (OCR layers)
    n_clipped_paths: int = 0  # paths dropped entirely by the CropBox / clip paths

    @property
    def empty(self) -> bool:
        """Nothing visible or searchable on the page."""
        return self.n_paths == 0 and self.n_chars == 0 and self.n_hidden_chars == 0 and self.image_coverage < 0.01

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
                    content.segments.append(Segment(a[0], a[1], b[0], b[1], width, color, dashed, pidx, sidx, closed, filled, -1, k, layer))

    # ---- text
    # TEXT_CLIP (== TEXT_MEDIABOX_CLIP in older releases): drop glyphs outside the page and clip paths
    clip_flag = getattr(pymupdf, "TEXT_CLIP", 0) | getattr(pymupdf, "TEXT_MEDIABOX_CLIP", 0)
    td = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES | clip_flag)
    page_box = (-0.5, -0.5, content.width + 0.5, content.height + 0.5)
    line_no = 0
    n_chars = n_hidden = 0
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
                hidden = span.get("alpha", 255) == 0
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


def _dedupe_words(words: list[TextWord]) -> list[TextWord]:
    """Drop a word drawn twice at the same place (duplicated CAD text entities)."""
    seen: set[tuple] = set()
    out = []
    for w in words:
        key = (w.text, w.rotation, w.hidden) + tuple(int(round(v * 10)) for v in w.bbox)
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


def open_pdf(path) -> "pymupdf.Document":
    """Open a PDF with PyMuPDF (raises ``ValueError`` for encrypted documents)."""
    doc = pymupdf.open(str(path))
    if doc.needs_pass:
        doc.close()
        raise ValueError(f"{path}: encrypted PDF")
    return doc
