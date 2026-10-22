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

Score = F1(precision, recall) x fill agreement, in [0, 1]. Two priors make the score
a plausibility rather than a pure shape fit:

* structure: a symbol must explain most of the candidate's strokes and the candidate
  must draw most of the symbol (``min(precision, recall) >= MIN_PR``), otherwise the
  score is cut by ``STRUCT_PENALTY`` (a box without the strainer's diagonal, hatching);
* size: stretchable equipment (tanks, engines ...) drawn smaller than a minimum size in
  symbol units is penalised (a small box between two valves is not a main engine).

Calibration on unseen symbol styles: the soft chamfer forgives a little extra ink (F1 of
precision 0.88 and recall 1.0 is 0.94), so a yard check valve drawn as a bow-tie plus an
outlet bar would match the gate template "confidently". Every fine match therefore also
measures the share of the candidate's ink (outline and filled regions) farther than
:data:`INK_DELTA` from the template -- ink on the through-axis between two opposite ports
excepted, where a pipe may be drawn through an inline symbol -- and the share of the
template's ink the candidate does not draw. Library drawings measure 0 for both; above a
small tolerance the score falls steeply (:func:`ink_penalty`), and a match that was
confident lands in :data:`REVIEW_BAND`: still the best guess, below the review threshold of
:class:`shippid.analysis.gemma.GemmaAnalyzer`. The same measure separates look-alike
library classes (gate vs ball valve, check vs SDNR valve) by clear margins.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
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
ANISO_MAX = 1.3  # rigid symbols may be drawn with proportions off by this factor (yard styles)
ANISO_PENALTY = 0.1  # score penalty at the maximum anisotropy (exact proportions preferred)
MIN_PR = 0.78  # structural floor on min(precision, recall); correct matches measure >= 0.83
STRUCT_PENALTY = 0.7  # score factor below the floor (keeps the class as a ranked alternative)
# minimum plausible size (max dimension, symbol units) of stretchable equipment; smaller is penalised
STRETCH_MIN_UNITS = {
    SymbolClass.TANK: 1.2, SymbolClass.ENGINE: 1.5, SymbolClass.AIR_RECEIVER: 1.1,
    SymbolClass.OILY_WATER_SEPARATOR: 1.0, SymbolClass.BWTS_UNIT: 1.1, SymbolClass.SEA_CHEST: 1.0,
}
STRETCH_MIN_DEFAULT = 1.0


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
    axes: tuple[tuple[float, float, float, float], ...] = ()  # through-axes between opposite ports (x0, y0, x1, y1)

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
    unexplained: float = 0.0  # share of the candidate's ink that the template does not explain
    missing: float = 0.0  # share of the template's ink that the candidate does not draw
    raw: Optional[float] = None  # score before the ink penalty (None: no penalty applied)

    def to_page(self, x: float, y: float) -> tuple[float, float]:
        return (self.cx + self.sx * x, self.cy + self.sy * y)

    @property
    def shape_score(self) -> float:
        """Shape fit without the unexplained / missing ink penalty (for scale estimation)."""
        return self.raw if self.raw is not None else self.score

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
        self.n_extra = 0  # leading variants added by :meth:`extended` (templates learned for one document)
        self._keys: list[tuple] = []  # dedupe key per variant
        self._add_specs(self.specs, set())
        self._build_arrays()

    @classmethod
    def extended(cls, extra: Sequence[SymbolSpec], base: Optional["TemplateLibrary"] = None) -> "TemplateLibrary":
        """``base`` (default: the cached library) with the variants of ``extra`` in front (they win
        ties), without rebuilding the base variants; used for templates learned from a drawing's
        legend (:mod:`shippid.extract.legend`)."""
        base = base if base is not None else get_library()
        lib = cls.__new__(cls)
        lib.specs = list(extra) + list(base.specs)
        lib.variants = []
        lib._keys = []
        seen: set = set()
        lib._add_specs(extra, seen)
        lib.n_extra = len(lib.variants)
        grids: list[Optional[np.ndarray]] = [None] * lib.n_extra
        for v, key in zip(base.variants, base._keys):
            if key in seen:
                continue
            seen.add(key)
            lib.variants.append(dataclasses.replace(v, idx=len(lib.variants)))
            lib._keys.append(key)
            grids.append(base.grid[v.idx])
        lib._build_arrays(grids)
        return lib

    def _add_specs(self, specs: Sequence[SymbolSpec], seen: set) -> None:
        for spec in specs:
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
                            self._keys.append(sig)

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
        axes = []
        for i, (_, ax, ay, adx, ady) in enumerate(ports):
            for _, bx, by, bdx, bdy in ports[i + 1:]:
                # opposite ports facing away from each other on one line: a pipe may be drawn through
                if abs(adx + bdx) < 1e-6 and abs(ady + bdy) < 1e-6 and abs((bx - ax) * ady - (by - ay) * adx) < 1e-6 \
                        and (bx - ax) * bdx + (by - ay) * bdy > 0:
                    axes.append((ax, ay, bx, by))
        v = Variant(-1, orig, rot, mir, act, pts, fill_pts, w, h, (cx, cy), (bx0 - cx, by0 - cy, bx1 - cx, by1 - cy),
                    ports, stripped, (fx0 - cx, fy0 - cy, fx1 - cx, fy1 - cy), n_circles=n_circ, axes=tuple(axes))
        v.tree = cKDTree(pts)
        v.fill_tree = cKDTree(fill_pts) if len(fill_pts) else None
        return v

    def _build_arrays(self, grids: Optional[Sequence[Optional[np.ndarray]]] = None) -> None:
        """Vectorised data of the coarse pass; ``grids``: distance grids already computed per variant."""
        V = len(self.variants)
        self.W = np.array([v.w for v in self.variants])
        self.H = np.array([v.h for v in self.variants])
        self.Dv = np.maximum(self.W, self.H)
        self.stretch = np.array([v.stretchable for v in self.variants])
        self.min_units = np.array([STRETCH_MIN_UNITS.get(v.cls, STRETCH_MIN_DEFAULT) if v.stretchable else 0.0
                                   for v in self.variants])
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
            pre = grids[v.idx] if grids is not None else None
            self.grid[v.idx] = pre if pre is not None else _dt_grid(v.pts / v.D, E)

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


ASPECT_STEP = 0.35  # log-aspect bucket of the cache key; rigid hits are re-validated at the new proportions


def _norm_axes(w: float, h: float, D: float) -> tuple[float, float]:
    """Per-axis normalisation lengths (thin shapes fall back to the max dimension)."""
    return (w if w >= 0.08 * D else D, h if h >= 0.08 * D else D)


def _prim_signature(prims, rings, bb, D: float) -> bytes:
    """Order-independent description of the strokes, normalised per axis to the bbox and
    quantised; drawings of one symbol that differ only by translation, scale or a small
    change of proportions (yard styles) share a key."""
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    nx, ny = _norm_axes(w, h, D)
    qx, qy = 48.0 / nx, 48.0 / ny
    items = []
    for p in prims:
        if p.kind == "circle":
            items.append(("c", round((p.cx - bb[0]) * qx), round((p.cy - bb[1]) * qy), round(p.r * qx), round(p.r * qy),
                          p.filled))
        else:
            a = (round((p.x0 - bb[0]) * qx), round((p.y0 - bb[1]) * qy))
            b = (round((p.x1 - bb[0]) * qx), round((p.y1 - bb[1]) * qy))
            items.append(("s",) + (a + b if a <= b else b + a) + (p.filled,))
    items.sort()
    aspect = round(math.log(max(w, 1e-3 * D) / max(h, 1e-3 * D)) / ASPECT_STEP)
    head = (aspect, w < 0.08 * D, h < 0.08 * D, sum(1 for r in rings if r.filled))
    return repr((head, items)).encode()


# --------------------------------------------------------------------------- matching


def _soft(d: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-np.square(d / tau))


def coarse_scores(lib: TemplateLibrary, cs: CandidateShape, unit: Optional[float] = None,
                  max_rigid: Optional[float] = None, size_prior: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores of every variant after bbox alignment. Returns (score, sx, sy).

    ``unit`` gates rigid symbols to a plausible scale (and, with ``size_prior``, applies
    the size prior of stretchable equipment); ``max_rigid``: candidates larger than this
    (points) are only compared with stretchable equipment templates."""
    V = len(lib)
    Wc, Hc = max(cs.w, 1e-6), max(cs.h, 1e-6)
    # transform: page = c_c + S * q  (q relative to the variant bbox centre)
    s_uni = (Wc + Hc) / (lib.W + lib.H)
    thin_w = lib.W < 0.08 * lib.Dv
    thin_h = lib.H < 0.08 * lib.Dv
    sx = np.where(~thin_w, Wc / np.maximum(lib.W, 1e-9), s_uni)
    sy = np.where(~thin_h, Hc / np.maximum(lib.H, 1e-9), s_uni)
    # rigid symbols: per-axis fit, anisotropy limited to ANISO_MAX
    geo = np.sqrt(sx * sy)
    ratio = np.clip(sx / sy, 1.0 / ANISO_MAX, ANISO_MAX)
    rigid = ~lib.stretch
    sx = np.where(rigid, geo * np.sqrt(ratio), sx)
    sy = np.where(rigid, geo / np.sqrt(ratio), sy)
    aniso_pen = np.where(rigid, 1.0 - ANISO_PENALTY * np.abs(np.log(ratio)) / math.log(ANISO_MAX), 1.0)
    # aspect gate for rigid symbols, plausible scale for all
    eps = 0.06
    asp_c = (Wc + eps * cs.D) / (Hc + eps * cs.D)
    asp_v = (lib.W + eps * lib.Dv) / (lib.H + eps * lib.Dv)
    gate = lib.stretch | (np.abs(np.log(asp_c / asp_v)) <= math.log(1.9))
    if unit:
        sc = np.sqrt(sx * sy)
        gate &= lib.stretch | ((sc >= 0.25 * unit) & (sc <= 5.0 * unit))
    if max_rigid is not None and cs.D > max_rigid:
        gate &= lib.stretch
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
    score[vsel] = f1 * fill_factor * aniso_pen[vsel]
    if unit and size_prior:
        score *= _size_factors(lib.min_units, cs.D, unit)
    return score, sx, sy


def _size_factors(min_units, D: float, unit: float):
    """Size prior of stretchable equipment: 1 above the class minimum, falling as
    (size / minimum) ** 1.5 below it (rigid templates: min_units 0 -> factor 1)."""
    m = np.asarray(min_units, dtype=float) * unit
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(m > 0, np.clip(D / np.where(m > 0, m, 1.0), 0.0, 1.0) ** 1.5, 1.0)
    return f


def size_factor(v: "Variant", D: float, unit: Optional[float]) -> float:
    if not unit or not v.stretchable:
        return 1.0
    m = STRETCH_MIN_UNITS.get(v.cls, STRETCH_MIN_DEFAULT) * unit
    return 1.0 if D >= m else (D / m) ** 1.5


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
        denx, deny = float((ac[:, 0] ** 2).sum()), float((ac[:, 1] ** 2).sum())
        nsx = float((ac[:, 0] * bc[:, 0]).sum()) / denx if denx > 1e-12 * len(a) * D * D else sx
        nsy = float((ac[:, 1] * bc[:, 1]).sum()) / deny if deny > 1e-12 * len(a) * D * D else sy
        nsx = min(max(nsx, 0.8 * sx0), 1.25 * sx0)
        nsy = min(max(nsy, 0.8 * sy0), 1.25 * sy0)
        if uniform:  # rigid symbol: keep the anisotropy within ANISO_MAX
            g = math.sqrt(nsx * nsy)
            r = min(max(nsx / nsy, 1.0 / ANISO_MAX), ANISO_MAX)
            nsx, nsy = g * math.sqrt(r), g / math.sqrt(r)
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
    unexplained, missing = _ink_shares(v, loc, d2, d1 / s_mean)
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
        # filled ink (a centre dot, a hatched triangle) counts like outline ink
        lim = INK_DELTA * D
        unexplained = max(unexplained, float((fd2 > lim).mean()))
        missing = max(missing, float((fd1 / s_mean > lim).mean()))
    score = f1 * (0.45 + 0.55 * fill)
    if uniform:
        score *= _aniso_factor(sx / sy)
    if min(precision, recall) < MIN_PR:
        score *= STRUCT_PENALTY
    penalised = ink_penalty(score, unexplained, missing)
    return MatchResult(v, float(penalised), sx, sy, cx, cy, precision, recall, fill, unexplained, missing,
                       float(score) if penalised < score else None)


INK_DELTA = 0.06  # a sample farther than this (relative to the template size) from the other drawing is not matched
INK_TOL = 0.015  # unexplained / missing ink share tolerated without penalty (library drawings measure 0)
INK_K = 10.0  # score factor 1 - INK_K * (excess share): 4.5 % unexplained ink costs a third of the score
REVIEW_BAND = (0.63, 0.73)  # a confident match with unexplained / missing ink keeps its rank inside this band


def ink_penalty(score: float, unexplained: float, missing: float) -> float:
    """Calibration on unseen symbol styles: a template that leaves part of the candidate's ink
    unexplained (a yard check valve = gate bow-tie + an outlet bar; a ball valve with a large
    filled ball) or that the candidate does not draw completely is not a confident match,
    however well the rest fits. The score falls by ``INK_K`` per unit of excess share; a
    match that was confident is mapped no lower than into :data:`REVIEW_BAND` (order kept,
    class kept, below the review threshold), so the class stays the best guess and the
    object goes to review."""
    excess = max(0.0, unexplained - INK_TOL) + max(0.0, missing - INK_TOL)
    if excess <= 0.0:
        return score
    out = score * max(0.0, 1.0 - INK_K * excess)
    if score >= REVIEW_BAND[0]:
        out = max(out, review_band(score))
    return min(out, score)


def review_band(score: float) -> float:
    """``score`` mapped linearly from [REVIEW_BAND[0], 1] into :data:`REVIEW_BAND`: the
    confidence of a match that has to be reviewed (order of the alternatives preserved)."""
    lo, hi = REVIEW_BAND
    if score <= lo:
        return score
    return lo + (hi - lo) * (min(score, 1.0) - lo) / (1.0 - lo)


def _seg_dist(p: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    ab = np.array((b[0] - a[0], b[1] - a[1]))
    L2 = float(ab @ ab)
    t = np.clip(((p - a) @ ab) / max(L2, 1e-12), 0.0, 1.0)
    return np.hypot(*(p - (np.asarray(a) + t[:, None] * ab)).T)


def _ink_shares(v: Variant, loc: np.ndarray, d_ct: np.ndarray, d_tc: np.ndarray) -> tuple[float, float]:
    """(unexplained, missing): the share of the candidate's outline samples farther than
    :data:`INK_DELTA` from the template -- not counting ink on a through-axis between two
    opposite ports, where a pipe may be drawn through an inline symbol -- and the share of
    the template's samples farther than that from the candidate (template units)."""
    lim = INK_DELTA * v.D
    far = d_ct > lim
    if v.axes and far.any():
        for x0, y0, x1, y1 in v.axes:
            far &= _seg_dist(loc, (x0, y0), (x1, y1)) > lim
    return float(far.mean()) if len(far) else 0.0, float((d_tc > lim).mean()) if len(d_tc) else 0.0


def _aniso_factor(ratio: float) -> float:
    """Score factor of a rigid symbol drawn with per-axis scale ratio ``ratio``."""
    return 1.0 - ANISO_PENALTY * min(1.0, abs(math.log(max(ratio, 1e-9))) / math.log(ANISO_MAX))


class MatchCache:
    """Reuses match results for candidates with the same normalised shape
    (P&IDs repeat a few symbol drawings hundreds of times)."""

    RES = 28

    def __init__(self, max_rigid: Optional[float] = None) -> None:
        self.max_rigid = max_rigid  # candidates larger than this (points) can only be stretchable equipment
        self._store: dict[bytes, list[tuple]] = {}
        self.hits = 0
        self.misses = 0

    def key(self, cs: CandidateShape, unit: Optional[float] = None) -> bytes:
        """Normalised geometry signature (primitives relative to the bbox, quantised), plus
        a half-octave size bucket relative to the symbol unit when one is known (the
        unit-dependent gates and priors pick different variants for a tiny and a large drawing)."""
        bucket = b"" if not unit else b"|%d" % round(2 * math.log2(max(cs.D, 1e-6) / unit))
        if cs.signature is not None:
            return cs.signature + bucket
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
        return aspect.tobytes() + np.packbits(occ).tobytes() + np.packbits(fo).tobytes() + bucket

    def get(self, k: bytes, cs: CandidateShape, lib: "TemplateLibrary") -> Optional[list[MatchResult]]:
        """Stored results re-aligned to this candidate. A key covers a range of proportions:
        rigid variants are re-checked at the new anisotropy (penalty updated, variants
        beyond ``ANISO_MAX`` dropped); if that removes the stored best, it is a miss."""
        rows = self._store.get(k)
        if rows is None:
            self.misses += 1
            return None
        nx, ny = _norm_axes(cs.w, cs.h, cs.D)
        c = cs.center
        out = []
        for n, (vi, sc, rsx, rsy, rcx, rcy, p, r, f, ratio0, ue, mi, raw) in enumerate(rows):
            v = lib.variants[vi]
            sx, sy = rsx * nx, rsy * ny
            if not v.stretchable:
                ratio = sx / sy if sy else math.inf
                if not (1.0 / (ANISO_MAX * 1.01) <= ratio <= ANISO_MAX * 1.01):
                    if n == 0:
                        self.misses += 1
                        return None
                    continue
                k = _aniso_factor(ratio) / _aniso_factor(ratio0)
                sc *= k
                raw = raw * k if raw is not None else None
            out.append(MatchResult(v, sc, sx, sy, c[0] + rcx * nx, c[1] + rcy * ny, p, r, f, ue, mi, raw))
        self.hits += 1
        return out

    def put(self, k: bytes, cs: CandidateShape, results: list[MatchResult]) -> None:
        """Store the alignments per axis relative to the bbox, so a hit with slightly
        different proportions gets a correspondingly stretched alignment."""
        nx, ny = _norm_axes(cs.w, cs.h, cs.D)
        c = cs.center
        self._store[k] = [(m.variant.idx, m.score, m.sx / nx, m.sy / ny, (m.cx - c[0]) / nx, (m.cy - c[1]) / ny,
                           m.precision, m.recall, m.fill, m.sx / m.sy if m.sy else 1.0, m.unexplained, m.missing,
                           m.raw)
                          for m in sorted(results, key=lambda m: -m.score)]


def match_candidate(cs: CandidateShape, lib: Optional[TemplateLibrary] = None, unit: Optional[float] = None,
                    top: int = FINE_TOP, cache: Optional[MatchCache] = None,
                    max_rigid: Optional[float] = None, size_prior: bool = False) -> list[MatchResult]:
    """Fine-matched results for the most promising variants, best first. ``unit`` is the
    drawing's symbol unit (points); ``size_prior`` applies the size prior of stretchable
    equipment relative to it (only meaningful when the unit is anchored by matched
    valves, not estimated from the very equipment being judged)."""
    lib = lib or get_library()
    if max_rigid is None and cache is not None:
        max_rigid = cache.max_rigid
    prior = bool(unit) and size_prior
    key = None
    results = None
    if cache is not None:
        key = cache.key(cs, unit)
        results = cache.get(key, cs, lib)
    if results is None:
        # picks are made without the size prior and the results are cached size-independent; box-like
        # (stretchable) classes get at most 2 of the ``top`` picks each and 4 together, so rigid
        # look-alikes are always evaluated and the size prior (below) can fall back on them
        score, sx, sy = coarse_scores(lib, cs, unit, max_rigid)
        order = np.argsort(-score)
        picked: list[int] = []
        per_cls: dict[int, int] = {}
        n_stretch = 0
        floor = max(0.05, 0.6 * float(score[order[0]])) if len(order) else 0.05
        for k in order:
            if score[k] <= floor:
                break
            c = int(lib.cls_idx[k])
            if lib.stretch[k]:
                if per_cls.get(c, 0) >= 2 or n_stretch >= 4:
                    continue
                n_stretch += 1
            elif per_cls.get(c, 0) >= 4:
                continue
            per_cls[c] = per_cls.get(c, 0) + 1
            picked.append(int(k))
            if len(picked) >= top:
                break
        results = [fine_match(lib.variants[k], cs, float(sx[k]), float(sy[k])) for k in picked]
        if cache is not None and key is not None:
            cache.put(key, cs, results)  # size-independent scores; the size prior is applied below
    if prior:
        for r in results:
            f = size_factor(r.variant, cs.D, unit)
            r.score *= f
            if r.raw is not None:
                r.raw *= f
    results.sort(key=lambda r: (-round(r.score, 4), r.variant.idx))
    if results:
        # among near-equal variants of the winning class prefer the canonical one (rotation 0, not mirrored):
        # e.g. a stretched rectangle fits rotation 0 and 90 equally
        best = results[0]
        slack = 0.02 if best.variant.stretchable else 0.01
        same = [r for r in results if r.variant.cls == best.variant.cls and r.score >= best.score - slack]
        canon = min(same, key=lambda r: r.variant.idx)
        if canon is not best:
            results.remove(canon)
            results.insert(0, canon)
    return results
