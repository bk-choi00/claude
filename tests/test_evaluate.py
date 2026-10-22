"""Tests for shippid.evaluate on hand-computed cases."""

from __future__ import annotations

import json

import pytest

from shippid.evaluate import evaluate_dataset, evaluate_document, match_boxes, norm_text, prf, write_eval_report
from shippid.model import (
    ActuatorType,
    BBox,
    Endpoint,
    PIDDocument,
    PIDObject,
    PipeRun,
    Point,
    SymbolClass,
    TextItem,
    TextRole,
)
from shippid.symbols import category_of

S = SymbolClass


def _o(oid, cls, box, tag=None, actuator=ActuatorType.NONE, page=0):
    return PIDObject(id=oid, cls=cls, category=category_of(cls), bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                     tag=tag, actuator=actuator, page=page)


def _p(pid, a, b, x0, x1, line=None, size=None):
    return PipeRun(id=pid, points=[Point(x=x0, y=5), Point(x=x1, y=5)], start=Endpoint(object_id=a),
                   end=Endpoint(object_id=b), line_number=line, size=size)


def _t(tid, text, box, role, owner=None):
    return TextItem(id=tid, text=text, bbox=BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]), role=role,
                    owner_id=owner)


@pytest.fixture
def gt() -> PIDDocument:
    return PIDDocument(source="gt", objects=[
        _o("A", S.TANK, (0, 0, 10, 10), tag="T-1"),
        _o("B", S.GATE_VALVE, (20, 0, 30, 10), tag="BA-V001", actuator=ActuatorType.HYDRAULIC),
        _o("C", S.PUMP_CENTRIFUGAL, (40, 0, 50, 10)),
        _o("D", S.CHECK_VALVE, (60, 0, 70, 10)),
    ], pipes=[
        _p("p1", "A", "B", 10, 20, "100A-BA-001", "100A"),
        _p("p2", "B", "C", 30, 40, "100A-BA-002", "100A"),
        _p("p3", "C", "D", 50, 60),
    ], texts=[
        _t("t1", "T-1", (0, -10, 10, -2), TextRole.TAG, "A"),
        _t("t2", "100A-BA-001", (10, 6, 20, 9), TextRole.LINE_NUMBER, "p1"),
        _t("t3", "NOTE 1", (0, 50, 30, 58), TextRole.NOTE),
    ])


@pytest.fixture
def pred() -> PIDDocument:
    return PIDDocument(source="pred", objects=[
        _o("a", S.TANK, (0, 0, 10, 10), tag="t-1"),  # same tag after normalisation
        _o("b", S.GLOBE_VALVE, (22, 0, 32, 10), tag="BA-V002", actuator=ActuatorType.HYDRAULIC),  # IoU 0.667
        _o("c", S.PUMP_CENTRIFUGAL, (45, 0, 55, 10)),  # IoU 0.333 -> unmatched
        _o("e", S.GATE_VALVE, (100, 0, 110, 10)),  # false positive
    ], pipes=[
        _p("q1", "a", "b", 10, 20, "100A-BA-001", "100A"),
        _p("q2", "b", "c", 30, 40, "100A-BA-003", "100A"),
        _p("q3", "c", "e", 70, 100),
    ], texts=[
        _t("s1", "T-1", (0, -10, 10, -2), TextRole.TAG, "a"),
        _t("s2", "100A-BA-001", (10, 6, 20, 9), TextRole.LABEL, "q1"),
        _t("s4", "EXTRA", (200, 200, 220, 206), TextRole.NOTE),
    ])


def test_prf_and_norm_text():
    assert prf(2, 2, 2) == {"tp": 2, "fp": 2, "fn": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert prf(0, 0, 0)["f1"] is None and prf(0, 3, 0)["f1"] == 0.0 and prf(0, 3, 0)["recall"] is None
    assert norm_text(" ba–v 001 ") == "BA-V001"


def test_hungarian_beats_greedy():
    gts = [BBox(x0=0, y0=0, x1=10, y1=10), BBox(x0=4, y0=0, x1=14, y1=10)]
    preds = [BBox(x0=1, y0=0, x1=11, y1=10), BBox(x0=-2, y0=0, x1=8, y1=10)]
    m = sorted(match_boxes(preds, gts, 0.5))
    assert [(p, g) for p, g, _ in m] == [(0, 1), (1, 0)]  # greedy would match only (0, 0)
    assert m[0][2] == pytest.approx(70 / 130) and m[1][2] == pytest.approx(80 / 120)
    assert match_boxes(preds, gts, 0.5, compatible=lambda p, g: g == 0) == [(0, 0, pytest.approx(90 / 110))]
    assert match_boxes([], gts, 0.5) == []


def test_document_detection_and_classes(gt, pred):
    r = evaluate_document(pred, gt)
    assert r["detection"] == prf(2, 2, 2)
    assert r["detection_class_aware"] == prf(1, 3, 3)
    pc = r["per_class"]
    assert pc["tank"] == {**prf(1, 0, 0), "support": 1}
    assert pc["gate_valve"] == {**prf(0, 1, 1), "support": 1}
    assert pc["globe_valve"] == {**prf(0, 1, 0), "support": 0}
    assert pc["pump_centrifugal"] == {**prf(0, 1, 1), "support": 1}
    assert pc["check_valve"] == {**prf(0, 0, 1), "support": 1}
    assert pc["check_valve"]["precision"] is None and pc["check_valve"]["recall"] == 0.0
    assert r["per_category"]["equipment"] == {**prf(1, 1, 1), "support": 2}
    assert r["per_category"]["valve"] == {**prf(1, 1, 1), "support": 2}
    assert r["classification"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert r["confusion"] == {"gate_valve": {"globe_valve": 1}, "tank": {"tank": 1}}


def test_document_tags_actuators_connections(gt, pred):
    r = evaluate_document(pred, gt)
    assert r["tag"] == {"correct": 1, "total": 2, "accuracy": 0.5, "missing": 0, "wrong": 1, "spurious": 0}
    assert r["actuator"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    c = r["connections"]
    assert (c["tp"], c["fp"], c["fn"]) == (1, 2, 2)
    assert c["precision"] == pytest.approx(1 / 3) and c["recall"] == pytest.approx(1 / 3)
    assert c["gt_pairs"] == 3 and c["pred_pairs"] == 3 and c["unmapped_pred_pairs"] == 2


def test_document_pipes_and_line_numbers(gt, pred):
    r = evaluate_document(pred, gt)
    assert r["line_number"] == {"correct": 1, "total": 2, "accuracy": 0.5, "matched": 2}
    assert r["pipe_size"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    pr = r["pipes"]["process"]
    assert pr["gt_length"] == 30 and pr["pred_length"] == 50
    assert pr["recall"] == pytest.approx(20 / 30) and pr["precision"] == pytest.approx(20 / 50)
    assert pr["f1"] == pytest.approx(0.5)
    assert r["pipes"]["signal"]["recall"] is None
    assert r["pipes"]["matched_gt_pipes"] == 2


def test_document_texts(gt, pred):
    t = evaluate_document(pred, gt)["texts"]
    assert (t["tp"], t["fp"], t["fn"]) == (2, 1, 1)
    assert t["role"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert t["content"]["accuracy"] == 1.0
    assert t["owner"] == {"correct": 2, "total": 2, "accuracy": 1.0}  # object and pipe owners mapped
    assert t["role_confusion"] == {"line_number": {"label": 1}, "tag": {"tag": 1}}


def test_identical_documents_score_perfectly(gt):
    r = evaluate_document(gt, gt)
    for key in ("detection", "detection_class_aware", "connections", "texts"):
        assert r[key]["f1"] == 1.0, key
    for key in ("tag", "actuator", "line_number", "pipe_size", "classification"):
        assert r[key]["accuracy"] == 1.0, key
    assert r["pipes"]["process"]["f1"] == pytest.approx(1.0)


def test_pages_are_matched_separately(gt):
    moved = gt.model_copy(deep=True)
    for o in moved.objects:
        o.page = 1
    r = evaluate_document(moved, gt)
    assert r["detection"]["tp"] == 0 and r["detection"]["fp"] == 4 and r["detection"]["fn"] == 4


def test_dataset_micro_macro_and_per_class(gt, pred):
    r = evaluate_dataset([(pred, gt), (gt, gt)])
    assert r["n_documents"] == 2 and r["level"] == "dataset"
    assert r["micro"]["detection"] == prf(6, 2, 2)
    assert r["macro"]["detection"]["f1"] == pytest.approx(0.75)
    assert r["micro"]["connections"]["f1"] == pytest.approx(4 / 6)
    assert r["macro"]["connections"]["f1"] == pytest.approx((1 / 3 + 1) / 2)
    assert r["micro"]["tag"] == {"correct": 3, "total": 4, "accuracy": 0.75}
    assert r["micro"]["line_number"]["accuracy"] == pytest.approx(3 / 4)
    assert r["per_class"]["tank"] == {**prf(2, 0, 0), "support": 2}
    assert r["per_class"]["gate_valve"] == {**prf(1, 1, 1), "support": 2}
    assert r["per_class"]["globe_valve"]["f1"] == 0.0
    assert r["macro"]["class_f1"] == pytest.approx((1.0 + 0.5 + 0.5 + 2 / 3) / 4)
    assert r["micro"]["pipes_process"]["recall"] == pytest.approx(50 / 60)
    assert r["confusion"]["gate_valve"] == {"globe_valve": 1, "gate_valve": 1}
    assert [d["source"] for d in r["documents"]] == ["gt", "gt"]
    assert r["documents"][0]["detection_f1"] == 0.5


def test_empty_dataset():
    r = evaluate_dataset([])
    assert r["n_documents"] == 0 and r["micro"]["detection"]["f1"] is None


def test_write_eval_report(tmp_path, gt, pred):
    ds = evaluate_dataset([(pred, gt), (gt, gt)])
    jp, mp = write_eval_report(ds, tmp_path / "eval")
    assert json.loads(jp.read_text(encoding="utf-8")) == json.loads(json.dumps(ds))
    md = mp.read_text(encoding="utf-8")
    assert "| detection f1 | 0.750 | 0.750 |" in md
    assert "| tank | 2 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |" in md
    assert "| globe_valve | 0 | 0 | 1 | 0 | 0.000 | - | 0.000 |" in md
    assert "## Documents" in md
    jp2, mp2 = write_eval_report(evaluate_document(pred, gt), tmp_path / "doc")
    md2 = mp2.read_text(encoding="utf-8")
    assert "| detection P / R / F1 | 0.500 / 0.500 / 0.500 |" in md2
    assert "| tag accuracy | 0.500 (1/2) |" in md2 and "## Per category" in md2
