import json

import pymupdf
import pytest
from pydantic import BaseModel

from shippid.catalog import classify_tag, detect_system, parse_line_number, pipe_od_m
from shippid.llm.base import LLMClient, LLMError, Message, MockBackend, schema_for
from shippid.llm.jsonutil import extract_json
from shippid.model import ActuatorType, BBox, PIDDocument, PIDObject, SymbolClass, SystemType, Category
from shippid.symbols import Placement, all_symbols, draw_symbol, get_symbol, placed_bbox, placed_ports


def test_every_class_has_symbol():
    defined = {s.cls for s in all_symbols()}
    missing = set(SymbolClass) - defined - {SymbolClass.UNKNOWN}
    assert not missing


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_ports_lie_on_symbol_bbox(rotation):
    for spec in all_symbols():
        pl = Placement(100, 100, scale=20, rotation=rotation if spec.rotatable else 0)
        x0, y0, x1, y1 = placed_bbox(spec, pl)
        for name, (x, y), _ in placed_ports(spec, pl):
            assert x0 - 1e-6 <= x <= x1 + 1e-6 and y0 - 1e-6 <= y <= y1 + 1e-6, (spec.cls, name)


def test_rotation_maps_flow_axis():
    pl = Placement(0, 0, scale=10, rotation=90)
    assert pl.to_page(1, 0) == pytest.approx((0, 10))
    pl = Placement(0, 0, scale=10, rotation=0, mirrored=True)
    assert pl.to_page(0, -1) == pytest.approx((0, 10))


def test_draw_symbol_produces_vector_paths():
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    shape = page.new_shape()
    draw_symbol(shape, get_symbol(SymbolClass.GLOBE_VALVE), Placement(200, 150, scale=20), ActuatorType.HANDWHEEL)
    shape.commit()
    drawings = page.get_drawings()
    assert len(drawings) >= 3
    assert any(d.get("fill") for d in drawings)  # filled centre dot


def test_bbox_iou():
    a = BBox(x0=0, y0=0, x1=10, y1=10)
    b = BBox(x0=5, y0=0, x1=15, y1=10)
    assert a.iou(b) == pytest.approx(50 / 150)


def test_document_roundtrip(tmp_path):
    doc = PIDDocument(objects=[PIDObject(id="o1", cls=SymbolClass.TANK, category=Category.EQUIPMENT,
                                         bbox=BBox(x0=0, y0=0, x1=1, y1=1))])
    p = doc.save(tmp_path / "d.json")
    assert PIDDocument.load(p) == doc


def test_catalog_helpers():
    assert parse_line_number("250A-BA-012-1S")["size"] == "250A"
    assert classify_tag("TCV-2101") == "instrument"
    assert classify_tag("BA-V012") == "valve"
    assert detect_system("FUEL OIL TRANSFER SYSTEM") == SystemType.FUEL_OIL_TRANSFER
    assert pipe_od_m("150A") == pytest.approx(0.1652)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('<think>{"no": 0}</think>```json\n{"a": [1,2,],}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! Here it is: {"b": "x}"} thanks') == {"b": "x}"}
    with pytest.raises(ValueError):
        extract_json("no json here")


class _Out(BaseModel):
    label: str
    score: float


def test_client_retries_with_feedback(tmp_path):
    answers = iter(["not json", json.dumps({"label": "pump", "score": 0.9})])
    backend = MockBackend(lambda m, s, n: next(answers))
    client = LLMClient(backend, cache_dir=tmp_path)
    out = client.generate_json([Message("user", "classify")], _Out)
    assert out.label == "pump"
    assert client.usage.calls == 2
    # the whole exchange (invalid answer + repair round) replays from cache without the backend
    backend2 = MockBackend(lambda m, s, n: (_ for _ in ()).throw(AssertionError("should be cached")))
    client2 = LLMClient(backend2, cache_dir=tmp_path)
    assert client2.generate_json([Message("user", "classify")], _Out).label == "pump"
    assert backend2.calls == []


def test_client_gives_up():
    client = LLMClient(MockBackend(lambda m, s, n: "{}"), max_attempts=2)
    with pytest.raises(LLMError):
        client.generate_json([Message("user", "x")], _Out)


def test_schema_has_no_refs():
    from shippid.model import AnalysisReport

    assert "$ref" not in json.dumps(schema_for(AnalysisReport))
