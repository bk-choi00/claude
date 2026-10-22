"""Scale-invariant template matching of symbol candidates against the library.

Every :class:`~shippid.symbols.SymbolSpec` (plus, for valves, every actuator
decoration) is rendered at scale 1 in each distinct orientation (rotation
0/90/180/270 x mirror, duplicates removed) and sampled into outline points and
filled-interior points. A candidate is compared with each variant after a
bounding-box similarity alignment (uniform scale; per-axis scale for
stretchable equipment):

* coarse pass, vectorised over all variants with distance-transform grids:
  soft symmetric chamfer (precision = candidate points near the template,
  recall = template points near the candidate) and fill presence;
* fine pass on the best variants: ICP refinement of scale/translation with
  KD-trees, exact soft chamfer on outlines and on filled regions.

Score = F1(precision, recall) x fill agreement, in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from ..model import ActuatorType, Category, SymbolClass
from ..symbols import (
    ACTUATOR_PRIMS,
    Circle,
    Line,
    Placement,
    Poly,
    Rect,
    SymbolSpec,
    all_symbols,
    placed_bbox,
    placed_ports,
    placed_prims,
    sample_outline,
)

GRID = 40  # distance-transform grid resolution
MARGIN = 0.3  # grid margin around the bbox, relative to the max dimension
TAU = 0.045  # soft chamfer scale, relative to the template's max dimension
TAU_FILL = 0.07
N_SAMPLES = 40  # outline samples per max dimension
N_COARSE = 20  # outline samples per max dimension in the coarse pass
MAX_COARSE_CAND = 90  # candidate points used in the coarse pass
FINE_TOP = 6


@dataclass
class Variant:
    idx: int
    spec: SymbolSpec
    rotation: int
    mirrored: bool
    actuator: ActuatorType
    pts: np.ndarray  # outline samples, template units, relative to the bbox centre
    fill_pts: np.ndarray  # filled-interior samples, same frame
    w: float
    h: float
    center: tuple[float, float]  # bbox centre in placement units (origin = placement centre)
    body: tuple[float, float, float, float]  # bbox without actuator, relative to the bbox centre
    ports: list[tuple[str, float, float, float, float]]  # name, x, y, dx, dy (relative to bbox centre)
    stripped: bool = False
    full: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # drawn bbox incl. stripped stubs, rel. to centre
    tree: Optional[cKDTree] = None
    fill_tree: Optional[cKDTree] = None
    n_circles: int = 0

    @property
    def cls(self) -> SymbolClass:
        return self.spec.cls

    @property
    def D(self) -> float:
        return max(self.w, self.h)

    @property
    def stretchable(self) -> bool:
        return self.spec.stretchable


@dataclass
class MatchResult:
    variant: Variant
    score: float
    sx: float  # page points per template unit along page x
    sy: float
    cx: float  # page position of the template bbox centre
    cy: float
    precision: float = 0.0
    recall: float = 0.0
    fill: float = 1.0

    def to_page(self, x: float, y: float) -> tuple[float, float]:
        return (self.cx + self.sx * x, self.cy + self.sy * y)

    @property
    def scale(self) -> float:
        return math.sqrt(abs(self.sx * self.sy))

    def bbox(self) -> tuple[float, float, float, float]:
        """Page bbox of the complete symbol as drawn (incl. actuator and port stubs)."""
        f = self.variant.full
        return (self.cx + self.sx * f[0], self.cy + self.sy * f[1], self.cx + self.sx * f[2], self.cy + self.sy * f[3])

    def body_bbox(self) -> tuple[float, float, float, float]:
        b = self.variant.body
        return (self.cx + self.sx * b[0], self.cy + self.sy * b[1], self.cx + self.sx * b[2], self.cy + self.sy * b[3])

    def ports(self) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        out = []
        for name, x, y, dx, dy in self.variant.ports:
            out.append((name, self.to_page(x, y), (dx, dy)))
        return out

    def placement(self) -> Placement:
        """Recover the :class:`~shippid.symbols.Placement` the symbol was drawn with."""
        v = self.variant
        ox, oy = self.to_page(-v.center[0], -v.center[1])
        if v.rotation in (90, 270):
            lsx, lsy = self.sy, self.sx
        else:
            lsx, lsy = self.sx, self.sy
        if v.stretchable:
            return Placement(ox, oy, scale=lsx, rotation=v.rotation, mirrored=v.mirrored, stretch=(1.0, lsy / lsx if lsx else 1.0))
        return Placement(ox, oy, scale=self.scale, rotation=v.rotation, mirrored=v.mirrored)


# --------------------------------------------------------------------------- sampling


def _fill_samples(prims, step: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in prims:
        if isinstance(p, Circle) and p.fill:
            n = max(2, int(p.r / step))
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    x, y = i * p.r / n, j * p.r / n
                    if x * x + y * y <= p.r * p.r * 1.0001:
                        out.append((p.cx + x, p.cy + y))
        elif isinstance(p, Poly) and p.fill:
            out.extend(_poly_fill(p.pts, step))
        elif isinstance(p, Rect) and p.fill:  # pragma: no cover - library rects are never filled
            out.extend(_poly_fill([(p.x0, p.y0), (p.x1, p.y0), (p.x1, p.y1), (p.x0, p.y1)], step))
    return out


def _poly_fill(pts: Sequence[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx = max(2, int((x1 - x0) / step) + 1)
    ny = max(2, int((y1 - y0) / step) + 1)
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    gx, gy = gx.ravel(), gy.ravel()
    inside = _points_in_poly(gx, gy, pts)
    sel = list(zip(gx[inside].tolist(), gy[inside].tolist()))
    sel.extend(sample_outline([Poly(tuple(pts), closed=True)], step))
    return sel


def _points_in_poly(x: np.ndarray, y: np.ndarray, pts: Sequence[tuple[float, float]]) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=bool)
    n = len(pts)
    for i in range(n):
        xa, ya = pts[i]
        xb, yb = pts[(i + 1) % n]
        cond = ((ya > y) != (yb > y)) & (x < (xb - xa) * (y - ya) / ((yb - ya) if yb != ya else 1e-12) + xa)
        inside ^= cond
    return inside


def _strip_port_stubs(spec: SymbolSpec) -> tuple:
    """Library primitives without straight stubs lying on a port's axis (they merge
    into the pipe when a pipe is connected, e.g. the overboard's inlet stub)."""
    keep = []
    for p in spec.prims:
        if isinstance(p, Line):
            drop = False
            for ps in spec.ports:
                for (ax, ay), (bx, by) in (((p.x0, p.y0), (p.x1, p.y1)), ((p.x1, p.y1), (p.x0, p.y0))):
                    if abs(ax - ps.x) < 1e-9 and abs(ay - ps.y) < 1e-9:
                        dx, dy = bx - ax, by - ay
                        n = math.hypot(dx, dy) or 1.0
                        if abs(dx / n + ps.dx) < 1e-6 and abs(dy / n + ps.dy) < 1e-6:
                            drop = True
            if drop:
                continue
        keep.append(p)
    return tuple(keep)


# --------------------------------------------------------------------------- library


class TemplateLibrary:
    """All distinct template variants of the current symbol library."""

    def __init__(self, specs: Optional[Sequence[SymbolSpec]] = None):
        self.specs = list(specs if specs is not None else all_symbols())
        self.variants: list[Variant] = []
        seen: set = set()
        for spec in self.specs:
            versions = [(spec, False)]
            stripped = _strip_port_stubs(spec)
            if len(stripped) != len(spec.prims) and stripped:
                versions.append((SymbolSpec(spec.cls, spec.category, spec.label_en, spec.label_ko, stripped, spec.ports,
                                            spec.boundary_ports, spec.stretchable, spec.holds_text, spec.rotatable,
                                            spec.description), True))
            rots = (0, 90, 180, 270) if spec.rotatable else (0,)
            mirs = (False, True) if spec.rotatable else (False,)
            acts = tuple(ACTUATOR_PRIMS) if spec.category == Category.VALVE else (ActuatorType.NONE,)
            for vspec, is_stripped in versions:
                for act in acts:
                    for rot in rots:
                        for mir in mirs:
                            v = self._make(vspec, rot, mir, act, is_stripped, spec)
                            if v is None:
                                continue
                            sig = (spec.cls, act, is_stripped, _signature(v))
                            if sig in seen:
                                continue
                            seen.add(sig)
                            v.idx = len(self.variants)
                            self.variants.append(v)
        self._build_arrays()

    @staticmethod
    def _make(vspec: SymbolSpec, rot: int, mir: bool, act: ActuatorType, stripped: bool, orig: SymbolSpec) -> Optional[Variant]:
        pl = Placement(0.0, 0.0, scale=1.0, rotation=rot, mirrored=mir)
        prims = placed_prims(vspec, pl, act)
        if not prims:
            return None
        x0, y0, x1, y1 = placed_bbox(vspec, pl, act)
        w, h = x1 - x0, y1 - y0
        D = max(w, h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        step = D / (N_SAMPLES * (2.5 if orig.stretchable else 1.0))  # stretched templates get sparse otherwise
        pts = np.asarray(sample_outline(prims, step), dtype=float) - (cx, cy)
        fill = _fill_samples(prims, step)
        fill_pts = np.asarray(fill, dtype=float).reshape(-1, 2) - (cx, cy) if fill else np.zeros((0, 2))
        bx0, by0, bx1, by1 = placed_bbox(orig, pl, ActuatorType.NONE)
        ports = [(n, px - cx, py - cy, dx, dy) for n, (px, py), (dx, dy) in placed_ports(orig, pl)]
        n_circ = sum(1 for p in prims if isinstance(p, Circle))
        fx0, fy0, fx1, fy1 = placed_bbox(orig, pl, act)
        v = Variant(-1, orig, rot, mir, act, pts, fill_pts, w, h, (cx, cy), (bx0 - cx, by0 - cy, bx1 - cx, by1 - cy),
                    ports, stripped, (fx0 - cx, fy0 - cy, fx1 - cx, fy1 - cy), n_circles=n_circ)
        v.tree = cKDTree(pts)
        v.fill_tree = cKDTree(fill_pts) if len(fill_pts) else None
        return v

    def _build_arrays(self) -> None:
        V = len(self.variants)
        self.W = np.array([v.w for v in self.variants])
        self.H = np.array([v.h for v in self.variants])
        self.Dv = np.maximum(self.W, self.H)
        self.stretch = np.array([v.stretchable for v in self.variants])
        self.has_fill = np.array([len(v.fill_pts) > 0 for v in self.variants])
        self.cls_idx = np.array([list(SymbolClass).index(v.cls) for v in self.variants])
        # the coarse pass uses every k-th outline sample (about N_COARSE per max dimension)
        coarse = [v.pts[:: max(1, int(round(N_SAMPLES * (2.5 if v.stretchable else 1.0) / N_COARSE)))] for v in self.variants]
        counts = [len(c) for c in coarse]
        self.pt_var = np.repeat(np.arange(V), counts)
        self.pts_all = np.concatenate(coarse)
        self.pt_count = np.array(counts, dtype=float)
        # distance grids in relative units (distance / D) over [-E/2, E/2]^2, E = 1 + 2*MARGIN
        E = 1.0 + 2 * MARGIN
        self.grid = np.zeros((V, GRID, GRID), dtype=np.float32)
        for v in self.variants:
            rel = v.pts / v.D
            self.grid[v.idx] = _dt_grid(rel, E)

    def __len__(self) -> int:
        return len(self.variants)


def _signature(v: Variant) -> tuple:
    q = np.round(v.pts / max(v.D, 1e-9), 2) + 0.0
    f = np.round(v.fill_pts / max(v.D, 1e-9), 1) + 0.0 if len(v.fill_pts) else np.zeros((0, 2))
    return (round(v.w, 3), round(v.h, 3), tuple(sorted(map(tuple, q.tolist()))), tuple(sorted(set(map(tuple, f.tolist())))))


def _dt_grid(rel_pts: np.ndarray, extent: float) -> np.ndarray:
    """Distance (in the same relative units) from each grid cell centre to the nearest point."""
    cell = extent / GRID
    occ = np.ones((GRID, GRID), dtype=bool)
    ij = np.floor((rel_pts + extent / 2) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < GRID) & (ij[:, 1] >= 0) & (ij[:, 1] < GRID)
    occ[ij[ok, 1], ij[ok, 0]] = False
    if occ.all():
        return np.full((GRID, GRID), extent, dtype=np.float32)
    return (distance_transform_edt(occ) * cell).astype(np.float32)


_LIB: Optional[TemplateLibrary] = None
_LIB_KEY: Optional[tuple] = None


def get_library() -> TemplateLibrary:
    """Cached library; rebuilt when :func:`shippid.symbols.register_symbol` changed it."""
    global _LIB, _LIB_KEY
    specs = all_symbols()
    key = tuple((s.cls, id(s)) for s in specs)
    if _LIB is None or key != _LIB_KEY:
        _LIB = TemplateLibrary(specs)
        _LIB_KEY = key
    return _LIB


# --------------------------------------------------------------------------- candidate geometry


@dataclass
class CandidateShape:
    """Sampled geometry of a symbol candidate (page coordinates)."""

    pts: np.ndarray
    fill_pts: np.ndarray
    bbox: tuple[float, float, float, float]
    n_circles: int = 0
    tree: Optional[cKDTree] = None
    fill_tree: Optional[cKDTree] = None
    signature: Optional[bytes] = None  # quantised primitive geometry (cache key)

    @property
    def w(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def D(self) -> float:
        return max(self.w, self.h, 1e-6)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def candidate_shape(prims, rings) -> CandidateShape:
    """Sample a set of :class:`~shippid.extract.primitives.Prim` (and the filled rings
    among them) the same way the templates are sampled."""
    bb = (min(p.bbox[0] for p in prims), min(p.bbox[1] for p in prims), max(p.bbox[2] for p in prims), max(p.bbox[3] for p in prims))
    D = max(bb[2] - bb[0], bb[3] - bb[1], 1e-6)
    step = D / N_SAMPLES
    pts: list[tuple[float, float]] = []
    fill: list[tuple[float, float]] = []
    n_circ = 0
    for p in prims:
        pts.extend(p.sample(step))
        if p.kind == "circle":
            n_circ += 1
            if p.filled:
                fill.extend(_fill_samples([Circle(p.cx, p.cy, p.r, fill=True)], step))
    for ring in rings:
        if ring.filled:
            fill.extend(_poly_fill(ring.pts, step))
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    farr = np.asarray(fill, dtype=float).reshape(-1, 2)
    cs = CandidateShape(arr, farr, bb, n_circ)
    cs.tree = cKDTree(arr)
    cs.fill_tree = cKDTree(farr) if len(farr) else None
    cs.signature = _prim_signature(prims, rings, bb, D)
    return cs


def _prim_signature(prims, rings, bb, D: float) -> bytes:
    """Order-independent, translation/scale-normalised description of the strokes;
    identical symbol drawings get identical keys despite float noise."""
    q = 48.0 / D
    items = []
    for p in prims:
        if p.kind == "circle":
            items.append(("c", round((p.cx - bb[0]) * q), round((p.cy - bb[1]) * q), round(p.r * q), p.filled))
        else:
            a = (round((p.x0 - bb[0]) * q), round((p.y0 - bb[1]) * q))
            b = (round((p.x1 - bb[0]) * q), round((p.y1 - bb[1]) * q))
            items.append(("s",) + (a + b if a <= b else b + a) + (p.filled,))
    items.sort()
    head = (round((bb[2] - bb[0]) / D, 2), round((bb[3] - bb[1]) / D, 2), sum(1 for r in rings if r.filled))
    return repr((head, items)).encode()


# --------------------------------------------------------------------------- matching


def _soft(d: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-np.square(d / tau))


def coarse_scores(lib: TemplateLibrary, cs: CandidateShape, unit: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores of every variant after bbox alignment. Returns (score, sx, sy)."""
    V = len(lib)
    Wc, Hc = max(cs.w, 1e-6), max(cs.h, 1e-6)
    # transform: page = c_c + S * q  (q relative to the variant bbox centre)
    s_uni = (Wc + Hc) / (lib.W + lib.H)
    thin_w = lib.W < 0.08 * lib.Dv
    thin_h = lib.H < 0.08 * lib.Dv
    sx = np.where(lib.stretch & ~thin_w, Wc / np.maximum(lib.W, 1e-9), s_uni)
    sy = np.where(lib.stretch & ~thin_h, Hc / np.maximum(lib.H, 1e-9), s_uni)
    # aspect gate for rigid symbols, plausible scale for all
    eps = 0.06
    asp_c = (Wc + eps * cs.D) / (Hc + eps * cs.D)
    asp_v = (lib.W + eps * lib.Dv) / (lib.H + eps * lib.Dv)
    gate = lib.stretch | (np.abs(np.log(asp_c / asp_v)) <= math.log(1.9))
    if unit:
        sc = np.sqrt(sx * sy)
        gate &= lib.stretch | ((sc >= 0.25 * unit) & (sc <= 5.0 * unit))
    score = np.zeros(V)
    vsel = np.nonzero(gate)[0]
    if len(vsel) == 0:
        return score, sx, sy
    s_mean = np.sqrt(sx * sy)
    E = 1.0 + 2 * MARGIN

    # template -> candidate (recall): look up the candidate's distance grid
    extent_c = E * cs.D
    cell_c = extent_c / GRID
    cgrid = _dt_grid((cs.pts - cs.center) / cs.D, E) * cs.D  # page units
    cx, cy = cs.center
    pmask = gate[lib.pt_var]
    pv = lib.pt_var[pmask]
    pts = lib.pts_all[pmask]
    gx = (sx[pv] * pts[:, 0] + extent_c / 2) / cell_c
    gy = (sy[pv] * pts[:, 1] + extent_c / 2) / cell_c
    ix = np.minimum(np.maximum(gx, 0), GRID - 1).astype(np.intp)
    iy = np.minimum(np.maximum(gy, 0), GRID - 1).astype(np.intp)
    excess = np.hypot(np.maximum(0, np.maximum(-gx, gx - GRID)), np.maximum(0, np.maximum(-gy, gy - GRID))) * cell_c
    d_tc = (cgrid[iy, ix] + excess) / (s_mean[pv] * lib.Dv[pv])
    recall = np.bincount(pv, weights=_soft(d_tc, TAU), minlength=V)[vsel] / lib.pt_count[vsel]

    # candidate -> template (precision): look up each variant's grid
    cpts = cs.pts if len(cs.pts) <= MAX_COARSE_CAND else cs.pts[:: int(math.ceil(len(cs.pts) / MAX_COARSE_CAND))]
    rel = cpts - cs.center  # (M, 2)
    cellv = E / GRID
    gx = (rel[None, :, 0] / (sx[vsel, None] * lib.Dv[vsel, None]) + E / 2) / cellv  # (Vs, M)
    gy = (rel[None, :, 1] / (sy[vsel, None] * lib.Dv[vsel, None]) + E / 2) / cellv
    ix = np.minimum(np.maximum(gx, 0), GRID - 1).astype(np.intp)
    iy = np.minimum(np.maximum(gy, 0), GRID - 1).astype(np.intp)
    excess = np.hypot(np.maximum(0, np.maximum(-gx, gx - GRID)), np.maximum(0, np.maximum(-gy, gy - GRID))) * cellv
    d_ct = lib.grid[vsel[:, None], iy, ix] + excess
    precision = _soft(d_ct, TAU).mean(axis=1)

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    c_fill = len(cs.fill_pts) > 0
    fill_factor = np.where(lib.has_fill[vsel] == c_fill, 1.0, 0.45)
    score[vsel] = f1 * fill_factor
    return score, sx, sy


def fine_match(v: Variant, cs: CandidateShape, sx: float, sy: float, iters: int = 3) -> MatchResult:
    """ICP-refined alignment and exact soft-chamfer score of one variant."""
    cx, cy = cs.center
    uniform = not v.stretchable
    sx0, sy0 = sx, sy
    D = v.D
    for _ in range(iters):
        page = np.column_stack((cx + sx * v.pts[:, 0], cy + sy * v.pts[:, 1]))
        d1, i1 = cs.tree.query(page)
        loc = np.column_stack(((cs.pts[:, 0] - cx) / sx, (cs.pts[:, 1] - cy) / sy))
        d2, i2 = v.tree.query(loc)
        lim = 0.15 * D * math.sqrt(sx * sy)
        a1 = v.pts[d1 <= lim]
        b1 = cs.pts[i1[d1 <= lim]]
        m2 = d2 * math.sqrt(sx * sy) <= lim
        a2 = v.pts[i2[m2]]
        b2 = cs.pts[m2]
        a = np.vstack((a1, a2))
        b = np.vstack((b1, b2))
        if len(a) < 6:
            break
        am, bm = a.mean(axis=0), b.mean(axis=0)
        ac, bc = a - am, b - bm
        if uniform:
            den = float((ac * ac).sum())
            if den <= 1e-12:
                break
            s = float((ac * bc).sum()) / den
            s = min(max(s, 0.8 * math.sqrt(sx0 * sy0)), 1.25 * math.sqrt(sx0 * sy0))
            nsx = nsy = s
        else:
            denx, deny = float((ac[:, 0] ** 2).sum()), float((ac[:, 1] ** 2).sum())
            nsx = float((ac[:, 0] * bc[:, 0]).sum()) / denx if denx > 1e-12 else sx
            nsy = float((ac[:, 1] * bc[:, 1]).sum()) / deny if deny > 1e-12 else sy
            nsx = min(max(nsx, 0.8 * sx0), 1.25 * sx0)
            nsy = min(max(nsy, 0.8 * sy0), 1.25 * sy0)
        ncx = bm[0] - nsx * am[0]
        ncy = bm[1] - nsy * am[1]
        done = abs(ncx - cx) + abs(ncy - cy) < 1e-3 * cs.D and abs(nsx - sx) < 1e-4 * sx
        cx, cy, sx, sy = ncx, ncy, nsx, nsy
        if done:
            break
    s_mean = math.sqrt(abs(sx * sy))
    page = np.column_stack((cx + sx * v.pts[:, 0], cy + sy * v.pts[:, 1]))
    d1, _ = cs.tree.query(page)
    recall = float(_soft(d1 / (s_mean * D), TAU).mean())
    loc = np.column_stack(((cs.pts[:, 0] - cx) / sx, (cs.pts[:, 1] - cy) / sy))
    d2, _ = v.tree.query(loc)
    precision = float(_soft(d2 / D, TAU).mean())
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # filled regions
    if v.fill_tree is None and cs.fill_tree is None:
        fill = 1.0
    elif v.fill_tree is None or cs.fill_tree is None:
        fill = 0.0
    else:
        fpage = np.column_stack((cx + sx * v.fill_pts[:, 0], cy + sy * v.fill_pts[:, 1]))
        fd1, _ = cs.fill_tree.query(fpage)
        floc = np.column_stack(((cs.fill_pts[:, 0] - cx) / sx, (cs.fill_pts[:, 1] - cy) / sy))
        fd2, _ = v.fill_tree.query(floc)
        fr = float(_soft(fd1 / (s_mean * D), TAU_FILL).mean())
        fp = float(_soft(fd2 / D, TAU_FILL).mean())
        fill = 2 * fr * fp / max(fr + fp, 1e-9)
    score = f1 * (0.45 + 0.55 * fill)
    return MatchResult(v, float(score), sx, sy, cx, cy, precision, recall, fill)


class MatchCache:
    """Reuses match results for candidates with the same normalised shape
    (P&IDs repeat a few symbol drawings hundreds of times)."""

    RES = 28

    def __init__(self) -> None:
        self._store: dict[bytes, list[tuple[int, float, float, float, float, float, float, float, float]]] = {}
        self.hits = 0
        self.misses = 0

    def key(self, cs: CandidateShape) -> bytes:
        """Normalised geometry signature: primitives relative to the bbox, quantised."""
        if cs.signature is not None:
            return cs.signature
        D = cs.D
        n = self.RES
        rel = (cs.pts - (cs.bbox[0], cs.bbox[1])) / D
        ij = np.minimum((rel * n).astype(int), n - 1)
        occ = np.zeros((n, n), dtype=np.uint8)
        occ[ij[:, 1], ij[:, 0]] = 1
        fo = np.zeros((n, n), dtype=np.uint8)
        if len(cs.fill_pts):
            frel = (cs.fill_pts - (cs.bbox[0], cs.bbox[1])) / D
            fij = np.clip((frel * n).astype(int), 0, n - 1)
            fo[fij[:, 1], fij[:, 0]] = 1
        aspect = np.array([round(cs.w / D, 2), round(cs.h / D, 2)], dtype=np.float32)
        return aspect.tobytes() + np.packbits(occ).tobytes() + np.packbits(fo).tobytes()

    def get(self, k: bytes, cs: CandidateShape, lib: "TemplateLibrary") -> Optional[list[MatchResult]]:
        rows = self._store.get(k)
        if rows is None:
            self.misses += 1
            return None
        self.hits += 1
        D = cs.D
        c = cs.center
        return [MatchResult(lib.variants[vi], sc, rsx * D, rsy * D, c[0] + rcx * D, c[1] + rcy * D, p, r, f)
                for vi, sc, rsx, rsy, rcx, rcy, p, r, f in rows]

    def put(self, k: bytes, cs: CandidateShape, results: list[MatchResult]) -> None:
        D = cs.D
        c = cs.center
        self._store[k] = [(m.variant.idx, m.score, m.sx / D, m.sy / D, (m.cx - c[0]) / D, (m.cy - c[1]) / D,
                           m.precision, m.recall, m.fill) for m in results]


def match_candidate(cs: CandidateShape, lib: Optional[TemplateLibrary] = None, unit: Optional[float] = None,
                    top: int = FINE_TOP, cache: Optional[MatchCache] = None) -> list[MatchResult]:
    """Fine-matched results for the most promising variants, best first."""
    lib = lib or get_library()
    key = None
    if cache is not None:
        key = cache.key(cs)
        hit = cache.get(key, cs, lib)
        if hit is not None:
            return hit
    score, sx, sy = coarse_scores(lib, cs, unit)
    order = np.argsort(-score)
    picked: list[int] = []
    per_cls: dict[int, int] = {}
    floor = max(0.05, 0.6 * float(score[order[0]])) if len(order) else 0.05
    for k in order[: 4 * top]:
        if score[k] <= floor:
            break
        c = int(lib.cls_idx[k])
        if per_cls.get(c, 0) >= 4:
            continue
        per_cls[c] = per_cls.get(c, 0) + 1
        picked.append(int(k))
        if len(picked) >= top:
            break
    results = [fine_match(lib.variants[k], cs, float(sx[k]), float(sy[k])) for k in picked]
    results.sort(key=lambda r: (-round(r.score, 4), r.variant.idx))
    if results:
        # among near-equal variants of the winning class prefer the canonical one (rotation 0, not mirrored):
        # e.g. a stretched rectangle fits rotation 0 and 90 equally
        top = results[0]
        slack = 0.02 if top.variant.stretchable else 0.01
        same = [r for r in results if r.variant.cls == top.variant.cls and r.score >= top.score - slack]
        canon = min(same, key=lambda r: r.variant.idx)
        if canon is not top:
            results.remove(canon)
            results.insert(0, canon)
    if cache is not None and key is not None:
        cache.put(key, cs, results)
    return results
