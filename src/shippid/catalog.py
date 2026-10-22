"""Marine domain catalogue shared by the generator, extractor, analysis and 3D stages.

Contents: ship piping systems (codes, names, media, keywords), ISO 14726 pipe
identification colours, tag / line-number patterns, JIS/ISO nominal pipe sizes
and typical physical dimensions of equipment for 3D layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .model import Medium, SymbolClass, SystemType


@dataclass(frozen=True)
class SystemInfo:
    system: SystemType
    code: str  # used in tags and line numbers
    name_en: str
    name_ko: str
    medium: Medium
    keywords: tuple[str, ...]  # title-block / note keywords used for system detection


SYSTEMS: dict[SystemType, SystemInfo] = {
    s.system: s
    for s in [
        SystemInfo(SystemType.BALLAST, "BA", "Ballast water system", "평형수 계통", Medium.BALLAST_WATER,
                   ("BALLAST", "BWTS", "W.B.T", "WBT", "평형수")),
        SystemInfo(SystemType.BILGE, "BG", "Bilge system", "빌지 계통", Medium.BILGE,
                   ("BILGE", "OILY WATER", "OWS", "빌지")),
        SystemInfo(SystemType.FUEL_OIL_TRANSFER, "FO", "Fuel oil transfer system", "연료유 이송 계통", Medium.FUEL_OIL,
                   ("F.O. TRANSFER", "FUEL OIL TRANSFER", "FO TRANSFER", "HFO", "연료유 이송")),
        SystemInfo(SystemType.FUEL_OIL_SUPPLY, "FS", "Fuel oil supply system", "연료유 공급 계통", Medium.FUEL_OIL,
                   ("F.O. SUPPLY", "FUEL OIL SUPPLY", "FO SUPPLY", "BOOSTER", "연료유 공급")),
        SystemInfo(SystemType.LUBE_OIL, "LO", "Lubricating oil system", "윤활유 계통", Medium.LUBE_OIL,
                   ("LUB", "L.O.", "LUBRICATING", "윤활유")),
        SystemInfo(SystemType.SEA_WATER_COOLING, "SW", "Sea water cooling system", "해수 냉각 계통", Medium.SEA_WATER,
                   ("SEA WATER", "S.W.", "COOLING S.W", "해수")),
        SystemInfo(SystemType.FRESH_WATER_COOLING, "CF", "Central fresh water cooling system", "청수 냉각 계통",
                   Medium.FRESH_WATER, ("FRESH WATER", "F.W.", "CENTRAL COOLING", "JACKET", "청수")),
        SystemInfo(SystemType.FIRE_MAIN, "FM", "Fire main system", "소화 계통", Medium.FIRE_WATER,
                   ("FIRE", "HYDRANT", "소화")),
        SystemInfo(SystemType.COMPRESSED_AIR, "CA", "Compressed air system", "압축공기 계통", Medium.COMPRESSED_AIR,
                   ("COMPRESSED AIR", "STARTING AIR", "AIR RECEIVER", "압축공기")),
        SystemInfo(SystemType.STEAM, "ST", "Steam & condensate system", "증기 계통", Medium.STEAM,
                   ("STEAM", "CONDENSATE", "BOILER", "증기")),
    ]
}

SYSTEM_BY_CODE: dict[str, SystemType] = {info.code: s for s, info in SYSTEMS.items()}


def detect_system(text: str) -> Optional[SystemType]:
    """Best keyword match of a title / free text to a system (None if no keyword hits)."""
    up = text.upper()
    best, best_score = None, 0
    for s, info in SYSTEMS.items():
        score = sum(len(k) for k in info.keywords if k.upper() in up)
        if score > best_score:
            best, best_score = s, score
    return best


# ISO 14726:2008 "Ships and marine technology - Identification colours for the content of
# piping systems": main colour per medium (RGB 0-1 for rendering).
MEDIUM_COLORS: dict[Medium, tuple[str, tuple[float, float, float]]] = {
    Medium.FRESH_WATER: ("blue", (0.11, 0.36, 0.75)),
    Medium.SEA_WATER: ("green", (0.13, 0.55, 0.25)),
    Medium.BALLAST_WATER: ("green", (0.30, 0.66, 0.38)),
    Medium.BILGE: ("black", (0.12, 0.12, 0.12)),
    Medium.FUEL_OIL: ("brown", (0.47, 0.27, 0.11)),
    Medium.LUBE_OIL: ("yellow", (0.93, 0.78, 0.12)),
    Medium.COMPRESSED_AIR: ("light blue", (0.55, 0.78, 0.93)),
    Medium.STEAM: ("silver grey", (0.70, 0.72, 0.75)),
    Medium.FIRE_WATER: ("red", (0.80, 0.12, 0.12)),
    Medium.UNKNOWN: ("grey", (0.55, 0.55, 0.55)),
}

# ISA 5.1 first / succeeding letters most used on ship P&IDs
ISA_FIRST_LETTER = {
    "P": "pressure", "T": "temperature", "L": "level", "F": "flow", "D": "density", "S": "speed",
    "V": "viscosity", "Q": "quality (e.g. oil content ppm)", "A": "analysis", "Z": "position",
}
ISA_FUNCTIONS = {
    "I": "indicator", "T": "transmitter", "S": "switch", "C": "controller", "V": "valve",
    "A": "alarm", "R": "recorder", "E": "element", "G": "gauge",
}

INSTRUMENT_TAG_RE = re.compile(r"^(?P<letters>[PTLFDSVQAZ][A-Z]{0,3})[-\s]?(?P<number>\d{2,5}[A-Z]?)$")
VALVE_TAG_RE = re.compile(
    r"^(?:(?P<sys1>[A-Z]{2})-V-?(?P<n1>\d{2,4}[A-Z]?)|V-?(?P<sys2>[A-Z]{2})-?(?P<n2>\d{2,4}[A-Z]?)|"
    r"(?P<sys3>[A-Z]{2})(?P<n3>\d{3})V)$"
)
EQUIPMENT_TAG_RE = re.compile(r"^(?P<sys>[A-Z]{2})-(?P<kind>[A-Z]{1,3})(?P<n>\d{1,3})(?P<suffix>[A-Z]?)$")
LINE_NUMBER_RE = re.compile(
    r"^(?:(?P<size1>\d{2,4})A?-(?P<sys1>[A-Z]{2})-(?P<n1>\d{3,4})(?:-(?P<spec1>[0-9A-Z]{1,4}))?|"
    r"(?P<sys2>[A-Z]{2})-(?P<n2>\d{3,4})-(?P<size2>\d{2,4})A)$"
)


def parse_line_number(text: str) -> Optional[dict[str, str]]:
    """Parse '250A-BA-012-1S' / '250-BA-012' / 'BA-012-250A' into size/system/number/spec."""
    m = LINE_NUMBER_RE.match(text.strip().upper())
    if not m:
        return None
    g = m.groupdict()
    if g["sys1"]:
        return {"size": f"{g['size1']}A", "system": g["sys1"], "number": g["n1"], "spec": g["spec1"] or ""}
    return {"size": f"{g['size2']}A", "system": g["sys2"], "number": g["n2"], "spec": ""}


def classify_tag(text: str) -> Optional[str]:
    """'instrument' | 'valve' | 'equipment' | 'line_number' | None for a text string."""
    t = text.strip().upper()
    if LINE_NUMBER_RE.match(t):
        return "line_number"
    if VALVE_TAG_RE.match(t):
        return "valve"
    if EQUIPMENT_TAG_RE.match(t):
        return "equipment"
    if INSTRUMENT_TAG_RE.match(t.replace(" ", "-")):
        return "instrument"
    return None


# Nominal pipe sizes (JIS 'A' designation == DN) -> outside diameter in mm (JIS G 3452/3454).
PIPE_OD_MM: dict[int, float] = {
    15: 21.7, 20: 27.2, 25: 34.0, 32: 42.7, 40: 48.6, 50: 60.5, 65: 76.3, 80: 89.1, 100: 114.3,
    125: 139.8, 150: 165.2, 200: 216.3, 250: 267.4, 300: 318.5, 350: 355.6, 400: 406.4, 450: 457.2,
    500: 508.0, 600: 609.6,
}


def pipe_od_m(size: Optional[str], default_mm: float = 114.3) -> float:
    """Outside diameter in metres from a size string like '150A', 'DN150', '150'."""
    if not size:
        return default_mm / 1000
    m = re.search(r"(\d{2,3})", size)
    if not m:
        return default_mm / 1000
    dn = int(m.group(1))
    if dn in PIPE_OD_MM:
        return PIPE_OD_MM[dn] / 1000
    nearest = min(PIPE_OD_MM, key=lambda k: abs(k - dn))
    return PIPE_OD_MM[nearest] / 1000


# Typical overall dimensions (length x width x height, metres) used as priors for the
# 3D planner (Gemma is asked to refine them; the heuristic planner uses them as-is).
TYPICAL_DIMENSIONS: dict[SymbolClass, tuple[float, float, float]] = {
    SymbolClass.TANK: (4.0, 3.0, 2.5),
    SymbolClass.PUMP_CENTRIFUGAL: (1.6, 0.9, 1.8),
    SymbolClass.PUMP_SCREW: (1.8, 0.7, 0.9),
    SymbolClass.HEAT_EXCHANGER: (3.0, 0.7, 0.9),
    SymbolClass.PLATE_HEAT_EXCHANGER: (1.4, 0.9, 2.2),
    SymbolClass.HEATER: (1.8, 0.5, 0.6),
    SymbolClass.FILTER: (0.5, 0.5, 0.8),
    SymbolClass.DUPLEX_FILTER: (0.9, 0.6, 1.1),
    SymbolClass.STRAINER: (0.6, 0.5, 0.6),
    SymbolClass.SEA_CHEST: (2.0, 1.5, 1.2),
    SymbolClass.OVERBOARD: (0.4, 0.4, 0.4),
    SymbolClass.PURIFIER: (1.6, 1.2, 1.7),
    SymbolClass.COMPRESSOR: (1.8, 1.1, 1.3),
    SymbolClass.AIR_RECEIVER: (1.2, 1.2, 3.2),
    SymbolClass.OILY_WATER_SEPARATOR: (1.8, 1.0, 1.9),
    SymbolClass.BWTS_UNIT: (2.4, 1.0, 1.6),
    SymbolClass.ENGINE: (12.0, 4.0, 10.0),
    SymbolClass.FLOW_METER: (0.5, 0.3, 0.4),
    SymbolClass.GATE_VALVE: (0.35, 0.3, 0.6),
    SymbolClass.GLOBE_VALVE: (0.35, 0.3, 0.6),
    SymbolClass.BALL_VALVE: (0.3, 0.3, 0.4),
    SymbolClass.BUTTERFLY_VALVE: (0.15, 0.4, 0.5),
    SymbolClass.CHECK_VALVE: (0.35, 0.3, 0.4),
    SymbolClass.SDNR_VALVE: (0.35, 0.3, 0.6),
    SymbolClass.THREE_WAY_VALVE: (0.4, 0.4, 0.7),
    SymbolClass.SAFETY_VALVE: (0.25, 0.25, 0.5),
    SymbolClass.INSTRUMENT_LOCAL: (0.15, 0.1, 0.15),
    SymbolClass.INSTRUMENT_PANEL: (0.15, 0.1, 0.15),
    SymbolClass.INSTRUMENT_SHARED: (0.15, 0.1, 0.15),
    SymbolClass.REDUCER: (0.25, 0.2, 0.2),
    SymbolClass.SPECTACLE_FLANGE: (0.06, 0.4, 0.6),
    SymbolClass.FLEXIBLE_HOSE: (0.4, 0.2, 0.2),
    SymbolClass.OFF_PAGE_CONNECTOR: (0.3, 0.3, 0.3),
    SymbolClass.UNKNOWN: (0.4, 0.4, 0.4),
}

# Engine-room deck levels above base line (metres) for a large ship (VLCC / large
# container ship order of magnitude); used as defaults by the 3D planners.
DEFAULT_DECKS: list[tuple[str, float]] = [
    ("tank_top", 2.5),
    ("floor", 4.5),
    ("3rd_deck", 10.5),
    ("2nd_deck", 16.0),
    ("upper_deck", 24.0),
]
