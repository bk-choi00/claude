"""Core data model shared by every stage of the pipeline.

All geometry is in PDF page space: points (1/72 inch), origin at the top-left
corner of the page, x to the right, y downward -- the convention PyMuPDF uses.
3D geometry (see :mod:`shippid.model3d`) is in metres with z up.

The same :class:`PIDDocument` type is used for synthetic ground truth, for the
deterministic vector extraction result and for the Gemma-refined result, so the
evaluator can compare any two of them directly.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- geometry


class Point(_Base):
    x: float
    y: float

    def dist(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class BBox(_Base):
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_points(cls, pts: Iterable[tuple[float, float]]) -> "BBox":
        xs, ys = zip(*pts)
        return cls(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Point:
        return Point(x=(self.x0 + self.x1) / 2, y=(self.y0 + self.y1) / 2)

    def expand(self, d: float) -> "BBox":
        return BBox(x0=self.x0 - d, y0=self.y0 - d, x1=self.x1 + d, y1=self.y1 + d)

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersects(self, other: "BBox") -> bool:
        return not (
            other.x0 > self.x1 or other.x1 < self.x0 or other.y0 > self.y1 or other.y1 < self.y0
        )

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        return self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol

    def iou(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


# --------------------------------------------------------------------------- taxonomy


class Category(str, Enum):
    EQUIPMENT = "equipment"
    VALVE = "valve"
    INSTRUMENT = "instrument"
    FITTING = "fitting"
    CONNECTOR = "connector"


class SymbolClass(str, Enum):
    """Canonical object classes. Geometry for each lives in :mod:`shippid.symbols`."""

    # equipment
    TANK = "tank"
    PUMP_CENTRIFUGAL = "pump_centrifugal"
    PUMP_SCREW = "pump_screw"
    HEAT_EXCHANGER = "heat_exchanger"
    PLATE_HEAT_EXCHANGER = "plate_heat_exchanger"
    HEATER = "heater"
    FILTER = "filter"
    DUPLEX_FILTER = "duplex_filter"
    STRAINER = "strainer"
    SEA_CHEST = "sea_chest"
    OVERBOARD = "overboard"
    PURIFIER = "purifier"
    COMPRESSOR = "compressor"
    AIR_RECEIVER = "air_receiver"
    OILY_WATER_SEPARATOR = "oily_water_separator"
    BWTS_UNIT = "bwts_unit"
    ENGINE = "engine"
    FLOW_METER = "flow_meter"
    # valves (body type; the actuator is an attribute)
    GATE_VALVE = "gate_valve"
    GLOBE_VALVE = "globe_valve"
    BALL_VALVE = "ball_valve"
    BUTTERFLY_VALVE = "butterfly_valve"
    CHECK_VALVE = "check_valve"
    SDNR_VALVE = "sdnr_valve"
    THREE_WAY_VALVE = "three_way_valve"
    SAFETY_VALVE = "safety_valve"
    # instruments
    INSTRUMENT_LOCAL = "instrument_local"
    INSTRUMENT_PANEL = "instrument_panel"
    INSTRUMENT_SHARED = "instrument_shared"
    # fittings
    REDUCER = "reducer"
    SPECTACLE_FLANGE = "spectacle_flange"
    FLEXIBLE_HOSE = "flexible_hose"
    # connectors
    OFF_PAGE_CONNECTOR = "off_page_connector"
    UNKNOWN = "unknown"


class ActuatorType(str, Enum):
    NONE = "none"
    HANDWHEEL = "handwheel"
    DIAPHRAGM = "diaphragm"  # pneumatic control valve
    HYDRAULIC = "hydraulic"  # remote-operated (e.g. ballast butterfly valves)
    ELECTRIC = "electric"
    QUICK_CLOSING = "quick_closing"  # SOLAS quick-closing valve on FO/LO tank outlets


class PipeKind(str, Enum):
    PROCESS = "process"  # solid line carrying a medium
    SIGNAL = "signal"  # dashed instrument / control signal line
    INSTRUMENT_TAP = "instrument_tap"  # short process connection to an instrument


class Medium(str, Enum):
    SEA_WATER = "sea_water"
    FRESH_WATER = "fresh_water"
    BALLAST_WATER = "ballast_water"
    BILGE = "bilge"
    FUEL_OIL = "fuel_oil"
    LUBE_OIL = "lube_oil"
    COMPRESSED_AIR = "compressed_air"
    STEAM = "steam"
    FIRE_WATER = "fire_water"
    UNKNOWN = "unknown"


class SystemType(str, Enum):
    BALLAST = "ballast"
    BILGE = "bilge"
    FUEL_OIL_TRANSFER = "fuel_oil_transfer"
    FUEL_OIL_SUPPLY = "fuel_oil_supply"
    LUBE_OIL = "lube_oil"
    SEA_WATER_COOLING = "sea_water_cooling"
    FRESH_WATER_COOLING = "fresh_water_cooling"
    FIRE_MAIN = "fire_main"
    COMPRESSED_AIR = "compressed_air"
    STEAM = "steam"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- objects


class Port(_Base):
    """A connection point of an object, in page coordinates."""

    name: str  # e.g. "in", "out", "suction", "discharge", "b0" (boundary nozzle)
    position: Point
    direction: Optional[Point] = None  # outward unit vector, if known


class PIDObject(_Base):
    id: str
    cls: SymbolClass
    category: Category
    bbox: BBox
    page: int = 0
    rotation: int = 0  # degrees, one of 0/90/180/270 (symbol local x-axis vs page x-axis)
    mirrored: bool = False
    tag: Optional[str] = None
    actuator: ActuatorType = ActuatorType.NONE
    ports: list[Port] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    text_ids: list[str] = Field(default_factory=list)  # associated TextItem ids
    confidence: float = 1.0
    source: str = "gt"  # "gt" | "vector" | "gemma" | "merged" | "manual"

    @property
    def center(self) -> Point:
        return self.bbox.center


class Junction(_Base):
    id: str
    position: Point
    page: int = 0
    kind: str = "tee"  # "tee" | "cross" | "corner" | "end"
    dotted: bool = False  # drawn with a connection dot


class Endpoint(_Base):
    """One end of a pipe run: an object port or a junction (exactly one is set),
    or a free end (both unset) for dangling lines."""

    object_id: Optional[str] = None
    port: Optional[str] = None
    junction_id: Optional[str] = None

    @property
    def node_id(self) -> Optional[str]:
        return self.object_id or self.junction_id


class PipeRun(_Base):
    """A pipe (or signal line) between two endpoints, as an orthogonal-ish polyline."""

    id: str
    page: int = 0
    points: list[Point]
    start: Endpoint
    end: Endpoint
    kind: PipeKind = PipeKind.PROCESS
    line_number: Optional[str] = None  # e.g. "150A-BA-001"
    size: Optional[str] = None  # nominal size as drawn, e.g. "150A" / "DN150"
    medium: Medium = Medium.UNKNOWN
    flow_direction: Optional[int] = None  # +1 start->end, -1 end->start, None unknown
    text_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source: str = "gt"

    @property
    def length(self) -> float:
        return sum(a.dist(b) for a, b in zip(self.points, self.points[1:]))


class TextRole(str, Enum):
    TAG = "tag"
    LINE_NUMBER = "line_number"
    INSTRUMENT = "instrument"  # text inside instrument bubble
    CONNECTOR = "connector"  # off-page connector text ("TO M/E", drawing ref)
    NOTE = "note"
    TITLE_BLOCK = "title_block"
    LEGEND = "legend"
    LABEL = "label"  # free equipment/system label
    UNKNOWN = "unknown"


class TextItem(_Base):
    id: str
    text: str
    bbox: BBox
    page: int = 0
    rotation: int = 0  # 0 or 90 etc. (reading direction, degrees counter-clockwise)
    size: float = 0.0  # font size in points
    role: TextRole = TextRole.UNKNOWN
    owner_id: Optional[str] = None  # object or pipe this text is attached to


class DrawingInfo(_Base):
    drawing_no: Optional[str] = None
    title: Optional[str] = None
    system: SystemType = SystemType.UNKNOWN
    systems: list[SystemType] = Field(default_factory=list)
    hull_no: Optional[str] = None
    ship_type: Optional[str] = None
    revision: Optional[str] = None
    sheet: Optional[str] = None
    scale: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PageInfo(_Base):
    """Page geometry. All coordinates in a PIDDocument are in the *displayed* page space
    (after the PDF /Rotate is applied), i.e. the space of ``page.rect`` and of rendered
    images; ``rotation`` records the page's /Rotate value."""

    index: int
    width: float
    height: float
    rotation: int = 0
    drawing_area: Optional[BBox] = None  # area inside the frame, excluding title block


class PIDDocument(_Base):
    schema_version: str = SCHEMA_VERSION
    source: str = ""  # file path or "synthetic:<seed>"
    pages: list[PageInfo] = Field(default_factory=list)
    info: DrawingInfo = Field(default_factory=DrawingInfo)
    objects: list[PIDObject] = Field(default_factory=list)
    junctions: list[Junction] = Field(default_factory=list)
    pipes: list[PipeRun] = Field(default_factory=list)
    texts: list[TextItem] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    # -- lookups -----------------------------------------------------------------
    def object_by_id(self) -> dict[str, PIDObject]:
        return {o.id: o for o in self.objects}

    def text_by_id(self) -> dict[str, TextItem]:
        return {t.id: t for t in self.texts}

    def junction_by_id(self) -> dict[str, Junction]:
        return {j.id: j for j in self.junctions}

    def objects_of(self, *categories: Category) -> list[PIDObject]:
        return [o for o in self.objects if o.category in categories]

    # -- io ----------------------------------------------------------------------
    def to_json(self, indent: Optional[int] = 1) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PIDDocument":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- analysis


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"


class Finding(_Base):
    id: str
    rule_id: str
    verdict: Verdict = Verdict.FAIL
    severity: Severity
    title: str
    description: str
    object_ids: list[str] = Field(default_factory=list)
    reference: Optional[str] = None  # e.g. "SOLAS II-2/4.2.2.3.4"
    source: str = "rule"  # "rule" | "gemma"
    confidence: float = 1.0


class FlowPath(_Base):
    source_id: str
    target_id: str
    node_ids: list[str]
    pipe_ids: list[str] = Field(default_factory=list)
    description: Optional[str] = None


class LLMUsage(_Base):
    backend: str = "none"
    model: Optional[str] = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)


class AnalysisReport(_Base):
    schema_version: str = SCHEMA_VERSION
    document: PIDDocument
    counts: dict[str, int] = Field(default_factory=dict)
    equipment_list: list[dict[str, Any]] = Field(default_factory=list)
    valve_list: list[dict[str, Any]] = Field(default_factory=list)
    instrument_list: list[dict[str, Any]] = Field(default_factory=list)
    line_list: list[dict[str, Any]] = Field(default_factory=list)
    flow_paths: list[FlowPath] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    summary_ko: str = ""
    summary_en: str = ""
    llm: LLMUsage = Field(default_factory=LLMUsage)
    meta: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisReport":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 3D layout


class Vec3(_Base):
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Deck(_Base):
    name: str  # e.g. "tank_top", "floor", "2nd_deck"
    z: float  # elevation of the deck plate, metres


class Placement3D(_Base):
    """Where a P&ID object sits in 3D. ``position`` is the centre of the object's
    footprint at its base (z = bottom of the object)."""

    object_id: str
    cls: SymbolClass
    position: Vec3
    yaw_deg: float = 0.0  # rotation about +z
    size: Vec3  # overall length (x), width (y), height (z) in metres before yaw
    deck: Optional[str] = None
    nozzles: dict[str, Vec3] = Field(default_factory=dict)  # port name -> world position
    rationale: Optional[str] = None


class PipeRoute3D(_Base):
    pipe_id: str
    points: list[Vec3]  # orthogonal polyline, metres
    diameter: float  # outside diameter, metres
    medium: Medium = Medium.UNKNOWN
    line_number: Optional[str] = None


class Layout3D(_Base):
    schema_version: str = SCHEMA_VERSION
    units: str = "m"
    planner: str = "heuristic"  # "gemma" | "heuristic"
    decks: list[Deck] = Field(default_factory=list)
    bounds_min: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))
    bounds_max: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=0))
    placements: list[Placement3D] = Field(default_factory=list)
    routes: list[PipeRoute3D] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    llm: LLMUsage = Field(default_factory=LLMUsage)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Layout3D":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def dump_json(obj: Any, path: str | Path) -> Path:
    """Write a plain JSON-serialisable object (or pydantic model) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, BaseModel):
        path.write_text(obj.model_dump_json(indent=1), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path
