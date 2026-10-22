"""Tests for shippid.analysis (graph, rules, lists, analyzer) on hand-built documents."""

from __future__ import annotations

from typing import Optional

import pytest

from analysis_builder import DocBuilder, pump_station
from shippid.analysis import rules as rules_mod
from shippid.analysis.analyzer import ReportReviewer, analyze, review_status, sorted_findings
from shippid.analysis.graph import (
    FlowNet,
    build_graph,
    components,
    effective_class,
    flow_paths,
    isolation_analysis,
    isolation_valves,
    object_adjacency,
    resolve_port,
    sources_and_sinks,
)
from shippid.analysis.lists import decode_isa, equipment_list, instrument_list, line_list, valve_list
from shippid.analysis.rules import RULES, drawing_systems, run_rules
from shippid.model import (
    ActuatorType,
    AnalysisReport,
    Category,
    Finding,
    Medium,
    PIDDocument,
    PipeKind,
    Point,
    Severity,
    SymbolClass,
    SystemType,
    TextRole,
    Verdict,
)

S = SymbolClass
A = ActuatorType
ST = SystemType
V = Verdict


def findings(doc: PIDDocument, rule_id: str) -> list[Finding]:
    return run_rules(doc, rule_ids=[rule_id])


def verdicts(doc: PIDDocument, rule_id: str) -> list[Verdict]:
    return [f.verdict for f in findings(doc, rule_id)]


# --------------------------------------------------------------------------- graph


def test_build_graph_nodes_edges_and_free_ends():
    b = DocBuilder()
    t = b.obj(S.TANK, 50, 100, name="NO.1 TK", tag="T-1")
    v = b.obj(S.GATE_VALVE, 150, 100, tag="V-1")
    j = b.junction(250, 100)
    p = b.obj(S.PUMP_CENTRIFUGAL, 350, 100)
    b.chain(t, v, j, p)
    b.pipe(j, None, b_xy=(250, 200), line="50A-BG-001", size="50A")
    doc = b.build()
    G = build_graph(doc)
    assert G.nodes[t]["kind"] == "object" and G.nodes[t]["cls"] == S.TANK and G.nodes[t]["tag"] == "T-1"
    assert G.nodes[t]["category"] == Category.EQUIPMENT and G.nodes[t]["label"] == "NO.1 TK"
    assert G.nodes[j]["kind"] == "junction"
    assert G.nodes["free:p4:end"]["kind"] == "free"
    assert G.number_of_edges() == 4
    data = G.edges[j, "free:p4:end", "p4"]
    assert data["pipe_id"] == "p4" and data["line_number"] == "50A-BG-001" and data["size"] == "50A"
    assert data["kind"] == PipeKind.PROCESS
    assert G.edges[j, p, "p3"]["end_port"] == "suction"


def test_missing_endpoint_reference_becomes_free_node():
    b = DocBuilder()
    v = b.obj(S.GATE_VALVE, 100, 100)
    b.pipe(v, None, b_xy=(200, 100))
    doc = b.build()
    doc.pipes[0].end.object_id = "o99"
    G = build_graph(doc)
    assert G.nodes["free:p1:end"]["ref"] == "o99"


def test_object_adjacency_collapses_junctions_and_ignores_signal_lines():
    b = DocBuilder()
    a = b.obj(S.GATE_VALVE, 100, 100)
    c = b.obj(S.GATE_VALVE, 300, 100)
    d = b.obj(S.GATE_VALVE, 200, 300)
    e = b.obj(S.GLOBE_VALVE, 400, 300, actuator=A.DIAPHRAGM)
    i = b.obj(S.INSTRUMENT_LOCAL, 200, 30)
    j1 = b.junction(200, 100)
    j2 = b.junction(200, 200)
    b.pipe(a, j1)
    b.pipe(j1, c)
    b.pipe(j1, j2)
    b.pipe(j2, d)
    b.pipe(i, j1, kind=PipeKind.INSTRUMENT_TAP)
    b.pipe(i, e, kind=PipeKind.SIGNAL)
    pairs = object_adjacency(b.build())
    expected = {frozenset(p) for p in [(a, c), (a, d), (c, d), (a, i), (c, i), (d, i)]}
    assert pairs == expected


def test_components_largest_first_with_isolated_objects():
    b = DocBuilder()
    x1 = b.obj(S.GATE_VALVE, 100, 100)
    x2 = b.obj(S.GATE_VALVE, 200, 100)
    x3 = b.obj(S.GATE_VALVE, 300, 100)
    y1 = b.obj(S.GATE_VALVE, 100, 300)
    y2 = b.obj(S.GATE_VALVE, 200, 300)
    lone = b.obj(S.FILTER, 500, 500)
    b.chain(x1, x2, x3)
    b.pipe(y1, y2)
    assert components(b.build()) == [[x1, x2, x3], [y1, y2], [lone]]


def test_port_inference_from_rotation():
    b = DocBuilder()
    cv = b.obj(S.CHECK_VALVE, 200, 100, rotation=180)
    o = b.objects[0]
    # rotation 180: local +x (outlet) points to page -x
    assert resolve_port(o, None, Point(x=190, y=100)) == "out"
    assert resolve_port(o, None, Point(x=210, y=100)) == "in"
    p = b.obj(S.PUMP_CENTRIFUGAL, 400, 100, rotation=90)
    po = b.objects[1]
    # rotation 90: suction (local -x) -> page -y; discharge (local -y) -> page +x
    assert resolve_port(po, None, Point(x=400, y=90)) == "suction"
    assert resolve_port(po, None, Point(x=410, y=100)) == "discharge"
    assert cv and p


def _cv_line(rotation: int, send_port: bool, cv_ports: tuple[str, str] = ("in", "out")) -> tuple[PIDDocument, dict]:
    b = DocBuilder()
    t1 = b.obj(S.TANK, 60, 200, name="NO.1 TK")
    p = b.obj(S.PUMP_SCREW, 200, 200)
    cv = b.obj(S.CHECK_VALVE, 320, 200, rotation=rotation)
    t2 = b.obj(S.TANK, 460, 200, name="NO.2 TK")
    b.pipe(t1, p)
    if send_port:
        b.pipe(p, (cv, cv_ports[0]))
        b.pipe((cv, cv_ports[1]), t2)
    else:
        b.pipe(p, cv, send_port=False)
        b.pipe(cv, t2, send_port=False)
    return b.build(), dict(t1=t1, p=p, cv=cv, t2=t2)


def test_flow_path_through_pump_and_check_valve():
    doc, ids = _cv_line(0, True)
    paths = flow_paths(doc)
    assert [(fp.source_id, fp.target_id) for fp in paths] == [(ids["t1"], ids["t2"])]
    fp = paths[0]
    assert fp.node_ids == [ids["t1"], ids["p"], ids["cv"], ids["t2"]]
    assert fp.pipe_ids == ["p1", "p2", "p3"]
    assert fp.description == "NO.1 TK -> pump_screw:o2 -> NO.2 TK"


def test_flow_path_blocked_by_reversed_check_valve_ports():
    doc, _ = _cv_line(0, True, cv_ports=("out", "in"))
    assert flow_paths(doc) == []


@pytest.mark.parametrize("rotation,expected", [(0, 1), (180, 0)])
def test_flow_path_check_valve_direction_from_rotation(rotation, expected):
    doc, _ = _cv_line(rotation, False)
    assert len(flow_paths(doc)) == expected


def test_flow_paths_respect_pipe_flow_direction():
    doc, _ = _cv_line(0, True)
    doc.pipes[0].flow_direction = -1  # drawn tank -> pump but flow marked pump -> tank
    assert flow_paths(doc) == []


def test_sources_sinks_and_connector_direction():
    b = DocBuilder()
    sc = b.obj(S.SEA_CHEST, 60, 100, name="LOW SEA CHEST")
    c_in = b.obj(S.OFF_PAGE_CONNECTOR, 60, 300, name="FROM S.W. CROSS-OVER", name_as_text=True)
    j = b.junction(150, 200)
    p = b.obj(S.PUMP_CENTRIFUGAL, 250, 200)
    j2 = b.junction(350, 200)
    ob = b.obj(S.OVERBOARD, 450, 100)
    c_out = b.obj(S.OFF_PAGE_CONNECTOR, 450, 300, name="TO WASH DECK")
    b.pipe(sc, j)
    b.pipe(c_in, j)
    b.chain(j, p, j2)
    b.pipe(j2, ob)
    b.pipe(j2, c_out)
    doc = b.build()
    sources, sinks = sources_and_sinks(doc)
    assert sources == [sc, c_in]
    assert sinks == [ob, c_out]
    assert len(flow_paths(doc)) == 4
    assert len(flow_paths(doc, max_paths=3)) == 3


def test_isolation_valves_pass_check_valves_and_stop_at_relief():
    b, ids = pump_station(nrv=(True, True), relief=(True, False))
    doc = b.build()
    p0 = ids["pumps"][0]
    assert isolation_valves(doc, None, p0) == [ids["suction_valves"][0], ids["discharge_valves"][0]]
    assert all(br.isolated for br in isolation_analysis(doc, None, p0))


def test_isolation_open_branch_reported():
    b = DocBuilder()
    sv = b.obj(S.GATE_VALVE, 100, 100)
    p = b.obj(S.PUMP_SCREW, 200, 100)
    h = b.obj(S.HEATER, 320, 100)
    b.chain(sv, p, h)
    doc = b.build()
    branches = isolation_analysis(doc, None, p)
    assert [br.valves for br in branches] == [[sv], []]
    assert branches[1].open_to == [h] and not branches[1].isolated


def test_effective_class_from_label_for_unknown_objects():
    b = DocBuilder()
    u = b.unknown(100, 100, name="NO.2 BILGE PUMP")
    o = b.objects[0]
    assert effective_class(o, "NO.2 BILGE PUMP") == S.PUMP_CENTRIFUGAL
    assert effective_class(o, "M/E L.O. SUMP TK") == S.TANK
    assert effective_class(o, "OWS FEED PUMP") == S.PUMP_CENTRIFUGAL
    assert effective_class(o, None) == S.UNKNOWN
    assert u


# --------------------------------------------------------------------------- systems / registry


def test_drawing_systems_basis():
    assert drawing_systems(DocBuilder(ST.BILGE).build()) == ([ST.BILGE], "drawing_info")
    assert drawing_systems(DocBuilder(title="FIRE MAIN SYSTEM").build()) == ([ST.FIRE_MAIN], "title")
    b = DocBuilder()
    v1, v2, v3 = (b.obj(S.GATE_VALVE, x, 100) for x in (100, 200, 300))
    b.pipe(v1, v2, line="250A-BA-001-1S")
    b.pipe(v2, v3, line="BA-002-250A")
    assert drawing_systems(b.build()) == ([ST.BALLAST], "line_numbers")
    assert drawing_systems(DocBuilder().build()) == ([], "unknown")


def test_all_rules_run_on_empty_document_with_stable_ids():
    out = run_rules(PIDDocument())
    assert [f.rule_id for f in out] == list(RULES)  # one finding per rule
    assert all(f.id == f"{f.rule_id}:1" for f in out)
    marine = [f for f in out if RULES[f.rule_id].group == "marine"]
    assert all(f.verdict in (V.NOT_APPLICABLE, V.INSUFFICIENT_DATA) for f in marine)
    assert len(RULES) >= 20


def test_rule_exception_becomes_insufficient_data(monkeypatch):
    def boom(ctx, r):
        raise RuntimeError("kaputt")

    rule = RULES["R_PUMP_ISOLATION"]
    monkeypatch.setitem(RULES, "R_PUMP_ISOLATION", rules_mod.Rule(rule.rule_id, rule.title, rule.title_ko,
                                                                  rule.severity, rule.reference, boom))
    out = run_rules(PIDDocument(), rule_ids=["R_PUMP_ISOLATION"])
    assert out[0].verdict == V.INSUFFICIENT_DATA and "kaputt" in out[0].description


# --------------------------------------------------------------------------- pump rules


def test_pump_discharge_nrv_pass_and_fail_on_parallel_pumps():
    b, ids = pump_station(nrv=(True, True))
    out = findings(b.build(), "R_PUMP_DISCHARGE_NRV")
    assert [f.verdict for f in out] == [V.PASS, V.PASS]
    assert out[0].object_ids == [ids["pumps"][0], ids["nrvs"][0]]
    assert out[0].reference and out[0].source == "rule"

    b, ids = pump_station(nrv=(True, False))
    out = findings(b.build(), "R_PUMP_DISCHARGE_NRV")
    assert [f.verdict for f in out] == [V.PASS, V.FAIL]
    assert out[1].severity == Severity.ERROR
    assert out[1].object_ids[:2] == [ids["pumps"][1], ids["pumps"][0]]
    assert "common line" in out[1].description


def test_pump_discharge_nrv_single_pump_not_applicable_and_reversed_fail():
    b, _ = pump_station(n_pumps=1, nrv=(False,))
    assert verdicts(b.build(), "R_PUMP_DISCHARGE_NRV") == [V.NOT_APPLICABLE]
    doc, _ = _cv_line(0, True, cv_ports=("out", "in"))
    out = findings(doc, "R_PUMP_DISCHARGE_NRV")
    assert out[0].verdict == V.FAIL and "reversed" in out[0].description


def test_pd_pump_relief_valve():
    b, ids = pump_station(pump_cls=S.PUMP_SCREW, relief=(True, False))
    out = findings(b.build(), "R_PD_PUMP_RELIEF")
    assert [f.verdict for f in out] == [V.PASS, V.FAIL]
    assert ids["reliefs"][0] in out[0].object_ids
    b, _ = pump_station(pump_cls=S.PUMP_CENTRIFUGAL)
    assert verdicts(b.build(), "R_PD_PUMP_RELIEF") == [V.NOT_APPLICABLE]


def _relief_line(rv_ports: tuple[str, str]) -> tuple[PIDDocument, str]:
    """Storage tank -> QC valve -> screw pump -> check valve -> settling tank, relief valve across the pump."""
    b = DocBuilder(ST.FUEL_OIL_TRANSFER)
    ja, jb = b.junction(180, 200), b.junction(320, 200)
    p = b.obj(S.PUMP_SCREW, 250, 200, name="TRANSFER PUMP")
    b.chain(b.obj(S.TANK, 60, 200, name="NO.1 H.F.O. STOR. TK"),
            b.obj(S.GATE_VALVE, 130, 200, tag="V1", actuator=A.QUICK_CLOSING), ja, p, jb,
            b.obj(S.CHECK_VALVE, 380, 200, tag="V2"), b.obj(S.TANK, 520, 200, name="NO.1 H.F.O. SETT. TK"))
    rv = b.obj(S.SAFETY_VALVE, 250, 140, tag="RV1")
    b.pipe(jb, (rv, rv_ports[0]))
    b.pipe((rv, rv_ports[1]), ja)
    return b.build(), rv


def test_pd_pump_relief_valve_drawn_reversed_fails():
    doc, rv = _relief_line(("in", "out"))
    f = findings(doc, "R_PD_PUMP_RELIEF")[0]
    assert f.verdict == V.PASS and rv in f.object_ids
    doc, rv = _relief_line(("out", "in"))  # outlet on the discharge junction, inlet on the suction junction
    f = findings(doc, "R_PD_PUMP_RELIEF")[0]
    assert f.verdict == V.FAIL and "drawn reversed" in f.description and rv in f.object_ids


def _screw_pumps_shared_suction(relief: bool, suction_cv: bool = False) -> tuple[PIDDocument, dict]:
    """Two screw pumps on one suction header, each discharging to its own consumer through a separate
    line (no common discharge), optionally with a relief valve back to its own suction."""
    b = DocBuilder(ST.LUBE_OIL)
    ids: dict = {"pumps": [], "reliefs": [], "cvs": []}
    hs = b.junction(150, 300)
    b.pipe(b.obj(S.TANK, 60, 300, name="L.O. SUMP TK"), hs)
    prev = hs
    for i in range(2):
        y = 300 + 150 * i
        js = b.junction(150, y) if i else hs
        if i:
            b.pipe(prev, js)
        ja, jb = b.junction(280, y), b.junction(420, y)
        p = b.obj(S.PUMP_SCREW, 350, y, name=f"NO.{i + 1} L.O. PUMP")
        chain: list = [js, b.obj(S.GATE_VALVE, 220, y, tag=f"V{i}1", actuator=A.HANDWHEEL)]
        if suction_cv and i == 1:
            cv = b.obj(S.CHECK_VALVE, 250, y, tag="CVB")
            chain.append(cv)
            ids["cvs"].append(cv)
        b.chain(*chain, ja, p, jb, b.obj(S.GLOBE_VALVE, 500, y, tag=f"V{i}3", actuator=A.HANDWHEEL),
                b.obj(S.ENGINE, 700, y, name=f"NO.{i + 1} G/E"))
        if relief and not (suction_cv and i == 1):
            rv = b.obj(S.SAFETY_VALVE, 350, y - 50, tag=f"RV{i}")
            b.pipe(jb, (rv, "in"))
            b.pipe((rv, "out"), ja)
            ids["reliefs"].append(rv)
        ids["pumps"].append(p)
        prev = js
    return b.build(), ids


@pytest.mark.parametrize("relief", [False, True])
def test_pump_discharge_nrv_relief_valve_back_to_suction_is_not_a_common_discharge(relief):
    doc, ids = _screw_pumps_shared_suction(relief)
    assert verdicts(doc, "R_PUMP_DISCHARGE_NRV") == [V.NOT_APPLICABLE, V.NOT_APPLICABLE]
    if relief:
        out = findings(doc, "R_PD_PUMP_RELIEF")
        assert [f.verdict for f in out] == [V.PASS, V.PASS]
        assert [ids["reliefs"][k] in out[k].object_ids for k in range(2)] == [True, True]


def test_pump_discharge_nrv_not_credited_from_another_pumps_suction():
    doc, ids = _screw_pumps_shared_suction(relief=True, suction_cv=True)
    out = findings(doc, "R_PUMP_DISCHARGE_NRV")
    assert [f.verdict for f in out] == [V.NOT_APPLICABLE, V.NOT_APPLICABLE]
    assert all(ids["cvs"][0] not in f.object_ids for f in out)


def test_pump_discharge_nrv_with_reliefs_on_common_discharge_header():
    b, ids = pump_station(pump_cls=S.PUMP_SCREW, relief=(True, True), nrv=(True, False))
    out = findings(b.build(), "R_PUMP_DISCHARGE_NRV")
    assert [f.verdict for f in out] == [V.PASS, V.FAIL] and ids["nrvs"][0] in out[0].object_ids
    assert "common line" in out[1].description
    assert verdicts(b.build(), "R_PD_PUMP_RELIEF") == [V.PASS, V.PASS]


def test_pump_isolation_rule():
    b, _ = pump_station()
    assert verdicts(b.build(), "R_PUMP_ISOLATION") == [V.PASS, V.PASS]
    b = DocBuilder()
    p = b.obj(S.PUMP_SCREW, 200, 100)
    b.chain(b.obj(S.GATE_VALVE, 100, 100), p, b.obj(S.HEATER, 320, 100))
    out = findings(b.build(), "R_PUMP_ISOLATION")
    assert out[0].verdict == V.FAIL and out[0].severity == Severity.INFO


# --------------------------------------------------------------------------- fuel oil


def _fo_transfer(qc: bool = True, system: ST = ST.FUEL_OIL_TRANSFER) -> tuple[PIDDocument, dict]:
    b = DocBuilder(system)
    stor = b.obj(S.TANK, 60, 200, name="NO.1 H.F.O. STOR. TK", stretch=(1.5, 1.0))
    qv = b.obj(S.GLOBE_VALVE, 170, 200, tag="FO-V001", actuator=A.QUICK_CLOSING if qc else A.HANDWHEEL)
    p = b.obj(S.PUMP_SCREW, 300, 200, name="NO.1 H.F.O. TRANSFER PUMP")
    cv = b.obj(S.CHECK_VALVE, 400, 200)
    sv = b.obj(S.GLOBE_VALVE, 500, 200, actuator=A.HANDWHEEL)
    sett = b.obj(S.TANK, 640, 200, name="NO.1 H.F.O. SETT. TK", stretch=(1.5, 1.0))
    b.chain(stor, qv, p, cv, sv, sett, medium=Medium.FUEL_OIL, line="100A-FO-001", size="100A")
    return b.build(), dict(stor=stor, qv=qv, sett=sett, p=p)


def test_quick_closing_valve_on_fo_tank_outlet():
    doc, ids = _fo_transfer(qc=True)
    out = findings(doc, "R_QC_VALVE_FO_LO_TANK")
    by_tank = {f.object_ids[0]: f for f in out}
    assert by_tank[ids["stor"]].verdict == V.PASS and ids["qv"] in by_tank[ids["stor"]].object_ids
    assert by_tank[ids["sett"]].verdict == V.NOT_APPLICABLE  # filling connection only
    assert "SOLAS II-2/4.2.2.3.4" in by_tank[ids["stor"]].reference

    doc, ids = _fo_transfer(qc=False)
    out = {f.object_ids[0]: f for f in findings(doc, "R_QC_VALVE_FO_LO_TANK")}
    assert out[ids["stor"]].verdict == V.FAIL and "not quick-closing" in out[ids["stor"]].description


def test_quick_closing_not_applicable_without_fo_lo_tanks():
    b, _ = pump_station(ST.BALLAST, source_name="NO.1 W.B.TK (P)", sink_name="NO.2 W.B.TK (S)")
    assert verdicts(b.build(), "R_QC_VALVE_FO_LO_TANK") == [V.NOT_APPLICABLE]


def _tank_outlet(name: str, actuator: ActuatorType) -> tuple[PIDDocument, str]:
    b = DocBuilder(ST.FUEL_OIL_TRANSFER)
    t = b.obj(S.TANK, 60, 200, name=name)
    v = b.obj(S.GATE_VALVE, 170, 200, actuator=actuator, tag="V1")
    b.chain(t, v, b.obj(S.PUMP_SCREW, 300, 200, name="F.O. TRANSFER PUMP"),
            b.obj(S.TANK, 500, 200, name="NO.2 H.F.O. SETT. TK"))  # filling connection only -> not applicable
    return b.build(), t


@pytest.mark.parametrize("name", ["NO.1 H.F.O. STOR. TK (D.B.)", "NO.1 H.F.O. DB TK", "F.O. DOUBLE BOTTOM TK (P)",
                                  "NO.2 F.O. TK (DB)", "H.F.O. D.B. TK (S)"])
def test_quick_closing_double_bottom_tanks_exempt(name):
    doc, t = _tank_outlet(name, A.HANDWHEEL)
    out = {f.object_ids[0]: f for f in findings(doc, "R_QC_VALVE_FO_LO_TANK")}
    assert out[t].verdict == V.NOT_APPLICABLE and "double-bottom" in out[t].description
    assert not any(f.verdict == V.FAIL for f in out.values())


@pytest.mark.parametrize("actuator,verdict", [(A.QUICK_CLOSING, V.PASS), (A.HYDRAULIC, V.PASS),
                                              (A.ELECTRIC, V.PASS), (A.DIAPHRAGM, V.PASS),
                                              (A.HANDWHEEL, V.FAIL), (A.NONE, V.FAIL)])
def test_quick_closing_accepts_remotely_operated_outlet_valves(actuator, verdict):
    doc, t = _tank_outlet("NO.1 H.F.O. SETT. TK", actuator)
    f = {f.object_ids[0]: f for f in findings(doc, "R_QC_VALVE_FO_LO_TANK")}[t]
    assert f.verdict == verdict
    if actuator in (A.HYDRAULIC, A.ELECTRIC, A.DIAPHRAGM):
        assert "remotely operated" in f.description and "verify" in f.description and f.confidence < 1.0
    elif actuator == A.QUICK_CLOSING:
        assert "quick-closing valve V1" in f.description and f.confidence == 1.0
    else:
        assert f"actuator: {actuator.value}" in f.description and f.severity == Severity.ERROR


def _service_tanks(*names: str) -> PIDDocument:
    b = DocBuilder(ST.FUEL_OIL_SUPPLY)
    for i, nm in enumerate(names):
        b.obj(S.TANK, 60, 100 + 120 * i, name=nm)
    return b.build()


def test_fo_service_tanks_min2():
    out = findings(_service_tanks("H.F.O. SERV. TK"), "R_FO_SERVICE_TANKS_MIN2")
    assert out[0].verdict == V.FAIL and out[0].object_ids == ["o1"]
    assert verdicts(_service_tanks("NO.1 H.F.O. SERV. TK", "NO.2 H.F.O. SERV. TK"),
                    "R_FO_SERVICE_TANKS_MIN2") == [V.PASS]
    assert verdicts(DocBuilder(ST.BALLAST).build(), "R_FO_SERVICE_TANKS_MIN2") == [V.NOT_APPLICABLE]


def test_fo_service_tanks_counted_per_fuel_type():
    # one HFO + one MDO service tank is not two per fuel type (SOLAS II-1/26.11): never a PASS
    f = findings(_service_tanks("H.F.O. SERV. TK", "M.D.O. SERV. TK"), "R_FO_SERVICE_TANKS_MIN2")[0]
    assert f.verdict == V.INSUFFICIENT_DATA and f.severity == Severity.WARNING
    assert "residual: H.F.O. SERV. TK" in f.description and "distillate: M.D.O. SERV. TK" in f.description
    assert "SC123" in f.description
    # standard arrangement: two residual service tanks + one distillate tank (cold start / repair)
    f = findings(_service_tanks("NO.1 H.F.O. SERV. TK", "NO.2 H.F.O. SERV. TK", "M.D.O. SERV. TK"),
                 "R_FO_SERVICE_TANKS_MIN2")[0]
    assert f.verdict == V.PASS and f.confidence == pytest.approx(0.7) and "cold starting" in f.description
    # one residual tank backed by two distillate tanks: UI SC123 equivalent arrangement (verify capacity)
    f = findings(_service_tanks("H.F.O. SERV. TK", "NO.1 M.G.O. SERV. TK", "NO.2 M.G.O. SERV. TK"),
                 "R_FO_SERVICE_TANKS_MIN2")[0]
    assert f.verdict == V.PASS and f.confidence == pytest.approx(0.6) and "equivalent arrangement" in f.description
    f = findings(_service_tanks("NO.1 H.F.O. SERV. TK", "NO.2 H.F.O. SERV. TK", "NO.1 M.D.O. SERV. TK",
                                "NO.2 M.D.O. SERV. TK"), "R_FO_SERVICE_TANKS_MIN2")[0]
    assert f.verdict == V.PASS and f.confidence == 1.0
    # the generator's three-tank variant: HFO + LSHFO are both residual fuel
    assert verdicts(_service_tanks("H.F.O. SERV. TK", "M.D.O. SERV. TK", "L.S.H.F.O. SERV. TK"),
                    "R_FO_SERVICE_TANKS_MIN2") == [V.PASS]
    # distillate-only ship
    assert verdicts(_service_tanks("M.G.O. SERV. TK"), "R_FO_SERVICE_TANKS_MIN2") == [V.FAIL]
    assert verdicts(_service_tanks("NO.1 M.G.O. DAY TK", "NO.2 M.G.O. DAY TK"),
                    "R_FO_SERVICE_TANKS_MIN2") == [V.PASS]


# --------------------------------------------------------------------------- bilge


def _bilge(n_pumps: int = 2, nrv_on_well2: bool = True, ows: str = "ok") -> tuple[PIDDocument, dict]:
    """Two bilge wells -> header -> n bilge pumps -> SDNR -> overboard; holding tank -> OWS feed pump ->
    OWS -> QIA -> 3-way (branch back to holding tank) -> SDNR -> overboard."""
    b = DocBuilder(ST.BILGE, title="BILGE SYSTEM")
    ids: dict = {}
    w1 = b.obj(S.TANK, 60, 100, name="BILGE WELL (P)")
    w2 = b.obj(S.TANK, 60, 250, name="BILGE WELL (S)")
    hs = b.junction(300, 100)
    hs2 = b.junction(300, 250)
    s1 = b.obj(S.STRAINER, 150, 100)
    n1 = b.obj(S.SDNR_VALVE, 220, 100, tag="BG-V001", actuator=A.HANDWHEEL)
    b.chain(w1, s1, n1, hs)
    s2 = b.obj(S.STRAINER, 150, 250)
    second = b.obj(S.SDNR_VALVE if nrv_on_well2 else S.GATE_VALVE, 220, 250, tag="BG-V002", actuator=A.HANDWHEEL)
    b.chain(w2, s2, second, hs2)
    b.pipe(hs, hs2)
    hd = b.junction(700, 100)
    pumps = []
    prev_s, prev_d = hs, hd
    for i in range(n_pumps):
        y = 100 + 80 * i
        js = b.junction(360, y + 40) if i else hs
        jd = b.junction(700, y + 40) if i else hd
        if i:
            b.pipe(prev_s, js)
            b.pipe(prev_d, jd)
        p = b.obj(S.PUMP_CENTRIFUGAL, 520, y + (40 if i else 0), name=f"NO.{i + 1} BILGE PUMP", tag=f"BG-P0{i + 1}")
        b.chain(js, b.obj(S.GLOBE_VALVE, 440, y + (40 if i else 0), tag=f"BG-V1{i}1", actuator=A.HANDWHEEL), p,
                b.obj(S.CHECK_VALVE, 600, y + (40 if i else 0), tag=f"BG-V1{i}2"), jd)
        pumps.append(p)
        prev_s, prev_d = js, jd
    ob_nrv = b.obj(S.SDNR_VALVE, 800, 100, tag="BG-V009", actuator=A.HANDWHEEL)
    ob = b.obj(S.OVERBOARD, 900, 100, name="OVERBOARD")
    b.chain(hd, ob_nrv, ob)
    # OWS line
    hold = b.obj(S.TANK, 60, 500, name="BILGE HOLDING TK")
    fp = b.obj(S.PUMP_SCREW, 250, 500, name="OWS FEED PUMP")
    rv = b.obj(S.SAFETY_VALVE, 250, 440)
    ja, jb = b.junction(200, 500), b.junction(300, 500)
    sep = b.obj(S.OILY_WATER_SEPARATOR, 450, 500, name="OILY WATER SEPARATOR", stretch=(1.2, 1.0))
    b.chain(hold, b.obj(S.GATE_VALVE, 140, 500, tag="BG-V020", actuator=A.HANDWHEEL), ja, fp, jb,
            b.obj(S.CHECK_VALVE, 350, 500, tag="BG-V021"), sep)
    b.pipe(jb, (rv, "in"))
    b.pipe((rv, "out"), ja)
    jq = b.junction(560, 500)
    qia = b.obj(S.INSTRUMENT_PANEL, 560, 440, tag="QIA-701", attrs={"letters": "QIA"})
    b.pipe(qia, jq, kind=PipeKind.INSTRUMENT_TAP)
    ob2 = b.obj(S.OVERBOARD, 900, 500, name="OVERBOARD (OWS)")
    ob2_nrv = b.obj(S.SDNR_VALVE, 800, 500, tag="BG-V023", actuator=A.HANDWHEEL)
    if ows == "no_3way":
        b.chain(sep, jq, ob2_nrv, ob2)
        tw = None
    else:
        tw = b.obj(S.THREE_WAY_VALVE, 660, 500, tag="BG-V022", actuator=A.ELECTRIC)
        b.chain(sep, jq, tw, ob2_nrv, ob2)
        if ows == "ok":
            rv2 = b.obj(S.GLOBE_VALVE, 660, 620, tag="BG-V024", actuator=A.HANDWHEEL)
            b.pipe((tw, "branch"), rv2)
            b.pipe(rv2, hold)
    if ows == "no_alarm":
        b.objects = [o for o in b.objects if o.id != qia]
        b.pipes = [p for p in b.pipes if p.start.object_id != qia]
    ids.update(w1=w1, w2=w2, second=second, pumps=pumps, sep=sep, tw=tw, hold=hold, ob=ob, ob2=ob2, feed=fp)
    return b.build(), ids


def test_bilge_suction_nrv():
    doc, ids = _bilge()
    out = findings(doc, "R_BILGE_SUCTION_NRV")
    assert [f.verdict for f in out] == [V.PASS, V.PASS]
    assert {f.object_ids[0] for f in out} == {ids["w1"], ids["w2"]}  # holding tank is not a bilge suction
    doc, ids = _bilge(nrv_on_well2=False)
    out = {f.object_ids[0]: f for f in findings(doc, "R_BILGE_SUCTION_NRV")}
    assert out[ids["w1"]].verdict == V.PASS
    assert out[ids["w2"]].verdict == V.FAIL and "SOLAS II-1/35-1" in out[ids["w2"]].reference


def test_bilge_pumps_min2():
    doc, ids = _bilge(n_pumps=2)
    out = findings(doc, "R_BILGE_PUMPS_MIN2")
    assert out[0].verdict == V.PASS and set(out[0].object_ids) == set(ids["pumps"])  # OWS feed pump excluded
    doc, _ = _bilge(n_pumps=1)
    assert verdicts(doc, "R_BILGE_PUMPS_MIN2") == [V.FAIL]


def test_bilge_pump_recognised_from_label_when_class_unknown():
    doc, ids = _bilge(n_pumps=2)
    o = next(o for o in doc.objects if o.id == ids["pumps"][1])
    o.cls = S.UNKNOWN  # extraction failed on this symbol; the name still says pump
    out = findings(doc, "R_BILGE_PUMPS_MIN2")
    assert out[0].verdict == V.PASS and ids["pumps"][1] in out[0].object_ids


def test_ows_three_way_valve_and_alarm():
    doc, ids = _bilge()
    out = findings(doc, "R_OWS_3WAY_RECIRC")
    assert out[0].verdict == V.PASS and ids["tw"] in out[0].object_ids and ids["hold"] in out[0].object_ids
    assert verdicts(doc, "R_OWS_15PPM_ALARM") == [V.PASS]
    doc, _ = _bilge(ows="no_3way")
    assert verdicts(doc, "R_OWS_3WAY_RECIRC") == [V.FAIL]
    doc, _ = _bilge(ows="no_recirc")
    out = findings(doc, "R_OWS_3WAY_RECIRC")
    assert out[0].verdict == V.FAIL and "does not lead back" in out[0].description
    doc, _ = _bilge(ows="no_alarm")
    assert verdicts(doc, "R_OWS_15PPM_ALARM") == [V.FAIL]


def test_overboard_valves_on_bilge_drawing_pass():
    doc, _ = _bilge()
    assert verdicts(doc, "R_OVERBOARD_VALVES") == [V.PASS, V.PASS]


def _overboard(valves: list[SymbolClass]) -> PIDDocument:
    b = DocBuilder(ST.SEA_WATER_COOLING)
    p = b.obj(S.PUMP_CENTRIFUGAL, 60, 100)
    nodes = [p] + [b.obj(c, 160 + 80 * k, 100, actuator=A.HANDWHEEL if c != S.CHECK_VALVE else A.NONE)
                   for k, c in enumerate(valves)]
    b.chain(*nodes, b.obj(S.OVERBOARD, 500, 100))
    return b.build()


@pytest.mark.parametrize("valves,verdict,text", [
    ([S.SDNR_VALVE], V.PASS, ""),
    ([S.BUTTERFLY_VALVE, S.CHECK_VALVE], V.PASS, ""),
    ([S.BUTTERFLY_VALVE], V.FAIL, "no non-return valve"),
    ([S.CHECK_VALVE], V.FAIL, "no positive-closing valve"),
    ([S.SAFETY_VALVE], V.FAIL, "no positive-closing valve"),
])
def test_overboard_valve_and_nrv(valves, verdict, text):
    out = findings(_overboard(valves), "R_OVERBOARD_VALVES")
    assert out[0].verdict == verdict and text in out[0].description


def test_overboard_bypass_route_without_nrv_fails():
    b = DocBuilder()
    p = b.obj(S.PUMP_CENTRIFUGAL, 60, 100)
    j1, j2 = b.junction(150, 100), b.junction(450, 100)
    b.pipe(p, j1)
    b.chain(j1, b.obj(S.SDNR_VALVE, 300, 100, actuator=A.HANDWHEEL), j2)
    b.chain(j1, b.obj(S.GATE_VALVE, 300, 200, actuator=A.HANDWHEEL), j2)  # by-pass around the SDNR
    b.pipe(j2, b.obj(S.OVERBOARD, 550, 100))
    out = findings(b.build(), "R_OVERBOARD_VALVES")
    assert out[0].verdict == V.FAIL and "no non-return valve" in out[0].description


# --------------------------------------------------------------------------- sea water / fire / air


def _sea_suction(n_chests: int, strainer: bool, system: ST, pump_name: str = "NO.{n} MAIN C.S.W. PUMP",
                 n_pumps: int = 2, nrv: bool = True) -> tuple[PIDDocument, dict]:
    b = DocBuilder(system)
    hs = b.junction(300, 100)
    chests = []
    for i in range(n_chests):
        y = 100 + 150 * i
        c = b.obj(S.SEA_CHEST, 60, y, name=["HIGH SEA CHEST", "LOW SEA CHEST"][i])
        nodes = [c, b.obj(S.BUTTERFLY_VALVE, 140, y, actuator=A.HANDWHEEL)]
        if strainer:
            nodes.append(b.obj(S.STRAINER, 220, y))
        j = hs if i == 0 else b.junction(300, y)
        b.chain(*nodes, j)
        if i:
            b.pipe(hs, j)
        chests.append(c)
    hd = b.junction(800, 100)
    pumps = []
    for i in range(n_pumps):
        y = 100 + 150 * i
        js = hs if i == 0 else b.junction(330, y + 20)
        jd = hd if i == 0 else b.junction(800, y + 20)
        if i:
            b.pipe(hs, js)
            b.pipe(hd, jd)
        p = b.obj(S.PUMP_CENTRIFUGAL, 500, y, name=pump_name.format(n=i + 1))
        chain = [js, b.obj(S.GATE_VALVE, 420, y, actuator=A.HANDWHEEL), p]
        if nrv:
            chain.append(b.obj(S.CHECK_VALVE, 600, y))
        b.chain(*chain, b.obj(S.GLOBE_VALVE, 700, y, actuator=A.HANDWHEEL), jd)
        pumps.append(p)
    b.chain(hd, b.obj(S.GLOBE_VALVE, 900, 100, actuator=A.HANDWHEEL),
            b.obj(S.OFF_PAGE_CONNECTOR, 1000, 100, name="TO FIRE HYDRANTS"))
    return b.build(), dict(chests=chests, pumps=pumps)


def test_sea_chests_min2_for_sw_cooling():
    doc, ids = _sea_suction(2, True, ST.SEA_WATER_COOLING)
    out = findings(doc, "R_SW_SEA_CHESTS_MIN2")
    assert out[0].verdict == V.PASS and out[0].object_ids == ids["chests"]
    doc, _ = _sea_suction(1, True, ST.SEA_WATER_COOLING)
    assert verdicts(doc, "R_SW_SEA_CHESTS_MIN2") == [V.FAIL]
    doc, _ = _sea_suction(1, True, ST.FIRE_MAIN)
    assert verdicts(doc, "R_SW_SEA_CHESTS_MIN2") == [V.NOT_APPLICABLE]


def test_strainer_before_pump():
    doc, ids = _sea_suction(2, True, ST.SEA_WATER_COOLING)
    assert verdicts(doc, "R_STRAINER_BEFORE_PUMP") == [V.PASS, V.PASS]
    doc, ids = _sea_suction(2, False, ST.SEA_WATER_COOLING)
    out = findings(doc, "R_STRAINER_BEFORE_PUMP")
    assert [f.verdict for f in out] == [V.FAIL, V.FAIL]
    assert set(out[0].object_ids[1:]) == set(ids["chests"])


def test_strainer_bypass_detected():
    doc, ids = _sea_suction(1, True, ST.SEA_WATER_COOLING, n_pumps=1)
    b = DocBuilder()
    b.objects, b.junctions, b.pipes = doc.objects, doc.junctions, doc.pipes
    strainer = next(o.id for o in doc.objects if o.cls == S.STRAINER)
    valve = next(o.id for o in doc.objects if o.cls == S.BUTTERFLY_VALVE)
    b.pipe((valve, "out"), "j1")  # pipe around the strainer
    doc.pipes = b.pipes
    out = findings(doc, "R_STRAINER_BEFORE_PUMP")
    assert out[0].verdict == V.FAIL and strainer not in out[0].object_ids


def test_fire_pumps_min2_and_discharge_nrv():
    doc, ids = _sea_suction(1, True, ST.FIRE_MAIN, pump_name="NO.{n} FIRE PUMP")
    assert verdicts(doc, "R_FIRE_PUMPS_MIN2") == [V.PASS]
    assert verdicts(doc, "R_FIRE_PUMP_DISCHARGE_NRV") == [V.PASS, V.PASS]
    # fire pumps are not double-reported by the generic pump rule
    assert verdicts(doc, "R_PUMP_DISCHARGE_NRV") == [V.NOT_APPLICABLE]
    doc, _ = _sea_suction(1, True, ST.FIRE_MAIN, pump_name="NO.{n} FIRE PUMP", n_pumps=1, nrv=False)
    assert verdicts(doc, "R_FIRE_PUMPS_MIN2") == [V.FAIL]
    assert verdicts(doc, "R_FIRE_PUMP_DISCHARGE_NRV") == [V.FAIL]


def _air(n_comp: int, n_recv: int, safety: bool) -> PIDDocument:
    b = DocBuilder(ST.COMPRESSED_AIR)
    hd = b.junction(300, 100)
    for i in range(n_comp):
        y = 100 + 100 * i
        c = b.obj(S.COMPRESSOR, 60, y, name=f"NO.{i + 1} MAIN AIR COMPRESSOR")
        j = hd if i == 0 else b.junction(300, y)
        b.chain(c, b.obj(S.CHECK_VALVE, 150, y), b.obj(S.GLOBE_VALVE, 220, y, actuator=A.HANDWHEEL), j)
        if i:
            b.pipe(hd, j)
    out = b.junction(800, 100)
    for i in range(n_recv):
        y = 100 + 150 * i
        r = b.obj(S.AIR_RECEIVER, 500, y, name=f"NO.{i + 1} MAIN AIR RECEIVER")
        js = hd if i == 0 else b.junction(330, y)
        jo = out if i == 0 else b.junction(800, y)
        if i:
            b.pipe(hd, js)
            b.pipe(out, jo)
        b.chain(js, b.obj(S.GLOBE_VALVE, 420, y, actuator=A.HANDWHEEL), r,
                b.obj(S.GLOBE_VALVE, 600, y, actuator=A.HANDWHEEL), jo)
        if safety:
            sv = b.obj(S.SAFETY_VALVE, 500, y - 60)
            b.pipe(r, (sv, "in"))
    b.chain(out, b.obj(S.GLOBE_VALVE, 900, 100, actuator=A.HANDWHEEL), b.obj(S.ENGINE, 1050, 100, name="M/E"))
    return b.build()


def test_starting_air_rules():
    doc = _air(2, 2, True)
    for rid in ("R_START_AIR_COMPRESSORS_MIN2", "R_START_AIR_RECEIVERS_MIN2"):
        assert verdicts(doc, rid) == [V.PASS]
    assert verdicts(doc, "R_AIR_RECEIVER_RELIEF") == [V.PASS, V.PASS]
    doc = _air(1, 1, False)
    for rid in ("R_START_AIR_COMPRESSORS_MIN2", "R_START_AIR_RECEIVERS_MIN2", "R_AIR_RECEIVER_RELIEF"):
        assert verdicts(doc, rid) == [V.FAIL]
    paths = flow_paths(_air(2, 2, True))
    assert {fp.source_id for fp in paths} == {"o1", "o4"} and all(fp.target_id == paths[0].target_id for fp in paths)


def test_system_specific_rules_not_applicable_on_other_systems():
    doc, _ = _sea_suction(2, True, ST.BALLAST, pump_name="NO.{n} BALLAST PUMP")
    for rid in ("R_BILGE_SUCTION_NRV", "R_BILGE_PUMPS_MIN2", "R_FIRE_PUMPS_MIN2", "R_FIRE_PUMP_DISCHARGE_NRV",
                "R_SW_SEA_CHESTS_MIN2", "R_OWS_3WAY_RECIRC", "R_OWS_15PPM_ALARM", "R_START_AIR_COMPRESSORS_MIN2",
                "R_START_AIR_RECEIVERS_MIN2", "R_AIR_RECEIVER_RELIEF", "R_QC_VALVE_FO_LO_TANK",
                "R_FO_SERVICE_TANKS_MIN2"):
        assert verdicts(doc, rid) == [V.NOT_APPLICABLE], rid
    assert verdicts(doc, "R_BILGE_PUMPS_MIN2") == [V.NOT_APPLICABLE]


def _bilge_tank_to_pump(tank_name: Optional[str], pump_name: str = "OWS FEED PUMP") -> tuple[PIDDocument, dict]:
    b = DocBuilder(ST.BILGE, title="BILGE SYSTEM")
    t = b.obj(S.TANK, 60, 200, name=tank_name) if tank_name else b.obj(S.TANK, 60, 200)
    p = b.obj(S.PUMP_SCREW, 300, 200, name=pump_name, tag="BG-P05")
    b.chain(t, b.obj(S.GATE_VALVE, 170, 200, tag="BG-V001", actuator=A.HANDWHEEL), p,
            b.obj(S.OILY_WATER_SEPARATOR, 500, 200, name="OILY WATER SEPARATOR"))
    return b.build(), dict(tank=t, pump=p)


@pytest.mark.parametrize("name", ["BILGE PRIMARY TK", "BILGE SETTLING TK", "BILGE WATER TK", "BILGE SEPARATED OIL TK",
                                  "BILGE HOLDING TK", "BILGE STORAGE TK"])
def test_oily_water_process_tanks_are_not_bilge_suctions(name):
    doc, ids = _bilge_tank_to_pump(name)
    for rid in ("R_BILGE_SUCTION_NRV", "R_STRAINER_BEFORE_PUMP"):
        out = findings(doc, rid)
        assert not any(f.verdict == V.FAIL for f in out), (rid, out)
        assert all(ids["tank"] not in f.object_ids for f in out)


@pytest.mark.parametrize("name,verdict", [("BILGE WELL (P)", V.FAIL), ("E/R BILGE (P)", V.FAIL),
                                          ("NO.1 HOLD BILGE", V.FAIL), ("DUCT KEEL BILGE WELL", V.FAIL),
                                          ("BILGE TK", V.INSUFFICIENT_DATA), (None, V.INSUFFICIENT_DATA)])
def test_bilge_suction_wells_positive_match_and_uncertain(name, verdict):
    doc, ids = _bilge_tank_to_pump(name, pump_name="NO.1 BILGE PUMP")
    f = findings(doc, "R_BILGE_SUCTION_NRV")[0]
    assert f.verdict == verdict and f.object_ids[0] == ids["tank"]
    if verdict == V.INSUFFICIENT_DATA:
        assert f.confidence <= 0.5 and "confirm whether it is a bilge suction" in f.description


def test_well_kind_classification():
    from shippid.analysis.rules import _well_kind

    assert _well_kind("BILGE WELL (S)", False) == "well"
    assert _well_kind("COFFERDAM", True) == "well" and _well_kind("COFFERDAM", False) is None
    assert _well_kind("AFT BILGE", False) == "well"  # compartment bilge, no tank word
    assert _well_kind("BILGE TK", True) == "uncertain"
    assert _well_kind("BILGE PRIMARY TK", True) is None and _well_kind("HOT WELL", True) is None
    assert _well_kind("NO.1 W.B.TK (P)", True) is None


def test_strainer_rule_unconfirmed_bilge_tank_is_insufficient_data():
    doc, ids = _bilge_tank_to_pump(None)  # unnamed tank -> gate valve -> OWS feed pump, no strainer
    assert verdicts(doc, "R_BILGE_SUCTION_NRV") == [V.INSUFFICIENT_DATA]
    f = findings(doc, "R_STRAINER_BEFORE_PUMP")[0]
    assert f.verdict == V.INSUFFICIENT_DATA and f.confidence <= 0.5 and "not confirmed" in f.description
    doc, ids = _bilge_tank_to_pump("BILGE WELL (P)", pump_name="NO.1 BILGE PUMP")
    f = findings(doc, "R_STRAINER_BEFORE_PUMP")[0]
    assert f.verdict == V.FAIL and ids["tank"] in f.object_ids


def test_starting_air_rules_ignore_emergency_service_and_control_units():
    b = DocBuilder(ST.COMPRESSED_AIR, title="STARTING AIR SYSTEM")
    c1 = b.obj(S.COMPRESSOR, 60, 100, name="MAIN AIR COMPRESSOR")
    c2 = b.obj(S.COMPRESSOR, 60, 300, name="EMERGENCY AIR COMPRESSOR")
    c3 = b.obj(S.COMPRESSOR, 60, 500, name="SERVICE AIR COMPRESSOR")
    r1 = b.obj(S.AIR_RECEIVER, 300, 100, name="MAIN AIR RECEIVER")
    r2 = b.obj(S.AIR_RECEIVER, 300, 300, name="CONTROL AIR RECEIVER")
    for c, r in ((c1, r1), (c2, r1), (c3, r2)):
        b.pipe(c, r)
    f = findings(b.build(), "R_START_AIR_COMPRESSORS_MIN2")[0]
    assert f.verdict == V.FAIL and f.object_ids[0] == c1
    assert "Not counted" in f.description and "EMERGENCY AIR COMPRESSOR" in f.description
    f = findings(b.build(), "R_START_AIR_RECEIVERS_MIN2")[0]
    assert f.verdict == V.FAIL and "CONTROL AIR RECEIVER" in f.description
    b.obj(S.COMPRESSOR, 60, 700, name="NO.2 MAIN AIR COMPRESSOR")
    b.obj(S.AIR_RECEIVER, 300, 500, name="NO.2 MAIN AIR RECEIVER")
    for rid in ("R_START_AIR_COMPRESSORS_MIN2", "R_START_AIR_RECEIVERS_MIN2"):
        f = findings(b.build(), rid)[0]
        assert f.verdict == V.PASS and f.description.startswith("2 main starting air"), rid
    b = DocBuilder(ST.COMPRESSED_AIR, title="CONTROL & SERVICE AIR SYSTEM")
    b.obj(S.COMPRESSOR, 60, 100, name="NO.1 SERVICE AIR COMPRESSOR")
    b.obj(S.COMPRESSOR, 60, 300, name="NO.2 SERVICE AIR COMPRESSOR")
    assert verdicts(b.build(), "R_START_AIR_COMPRESSORS_MIN2") == [V.INSUFFICIENT_DATA]


def test_bilge_drawing_hot_well_is_not_a_bilge_suction():
    b = DocBuilder(ST.STEAM)
    hw = b.obj(S.TANK, 60, 100, name="HOT WELL")
    b.chain(hw, b.obj(S.GATE_VALVE, 150, 100), b.obj(S.PUMP_CENTRIFUGAL, 250, 100))
    assert verdicts(b.build(), "R_BILGE_SUCTION_NRV") == [V.NOT_APPLICABLE]


# --------------------------------------------------------------------------- drawing quality


def test_quality_unconnected_dangling_duplicates_missing_tags():
    b = DocBuilder(ST.BALLAST)
    v1 = b.obj(S.GATE_VALVE, 100, 100, tag="BA-V001")
    v2 = b.obj(S.GATE_VALVE, 200, 100, tag="BA-V001")
    lone = b.obj(S.FILTER, 400, 400)
    i = b.obj(S.INSTRUMENT_LOCAL, 300, 300)
    b.pipe(v1, v2)
    b.pipe(v2, None, b_xy=(300, 100))
    b.pipe(i, v2, kind=PipeKind.INSTRUMENT_TAP)
    doc = b.build()
    out = findings(doc, "DQ_UNCONNECTED_OBJECTS")
    assert out[0].verdict == V.FAIL and out[0].object_ids == [lone]
    out = findings(doc, "DQ_DANGLING_PIPE_ENDS")
    assert out[0].verdict == V.FAIL and out[0].object_ids == ["p2"]
    out = findings(doc, "DQ_DUPLICATE_TAGS")
    assert out[0].verdict == V.FAIL and out[0].object_ids == [v1, v2]
    out = findings(doc, "DQ_MISSING_TAGS")
    assert out[0].verdict == V.FAIL and set(out[0].object_ids) == {lone, i}
    out = findings(doc, "DQ_OPEN_INLINE_ITEMS")
    assert out[0].verdict == V.FAIL and out[0].object_ids == [v1]


def test_quality_checks_pass_on_clean_station():
    b, _ = pump_station(ST.BALLAST)
    doc = b.build()
    for p in doc.pipes:
        p.line_number = "250A-BA-001"
    for rid in ("DQ_UNCONNECTED_OBJECTS", "DQ_DANGLING_PIPE_ENDS", "DQ_DUPLICATE_TAGS", "DQ_OPEN_INLINE_ITEMS",
                "DQ_MISSING_TAGS", "DQ_LINE_SYSTEM_CODE", "DQ_SIZE_CHANGE_NO_REDUCER"):
        assert verdicts(doc, rid) == [V.PASS], rid


def test_size_change_without_reducer():
    b = DocBuilder()
    v1, v2, v3 = b.obj(S.GATE_VALVE, 100, 100), b.obj(S.GATE_VALVE, 200, 100), b.obj(S.GATE_VALVE, 300, 100)
    b.pipe(v1, v2, size="100A")
    b.pipe(v2, v3, size="80A")
    out = findings(b.build(), "DQ_SIZE_CHANGE_NO_REDUCER")
    assert out[0].verdict == V.FAIL and out[0].object_ids == [v2, "p1", "p2"] and "100A / 80A" in out[0].description
    b = DocBuilder()
    v1, r, v3 = b.obj(S.GATE_VALVE, 100, 100), b.obj(S.REDUCER, 200, 100), b.obj(S.GATE_VALVE, 300, 100)
    b.pipe(v1, r, size="DN100")
    b.pipe(r, v3, size="80A")
    b.pipe(b.obj(S.GATE_VALVE, 100, 300), b.obj(S.GATE_VALVE, 200, 300), size="100A")
    assert verdicts(b.build(), "DQ_SIZE_CHANGE_NO_REDUCER") == [V.PASS]


def test_size_change_on_tee_only_flagged_on_the_same_line():
    def tee(run_lines: tuple[str, str]) -> PIDDocument:
        b = DocBuilder()
        j = b.junction(200, 100)
        b.pipe(b.obj(S.GATE_VALVE, 100, 100), j, line=run_lines[0])
        b.pipe(j, b.obj(S.GATE_VALVE, 300, 100), line=run_lines[1])
        b.pipe(j, b.obj(S.GATE_VALVE, 200, 200), line="50A-BA-009")
        return b.build()

    assert verdicts(tee(("350A-BA-001", "250A-BA-002")), "DQ_SIZE_CHANGE_NO_REDUCER") == [V.PASS]
    assert verdicts(tee(("350A-BA-001", "250A-BA-001")), "DQ_SIZE_CHANGE_NO_REDUCER") == [V.FAIL]


def test_line_system_code_mismatch_and_interface_lines():
    b = DocBuilder(ST.BALLAST)
    v1, v2, v3 = b.obj(S.GATE_VALVE, 100, 100), b.obj(S.GATE_VALVE, 200, 100), b.obj(S.GATE_VALVE, 300, 100)
    b.pipe(v1, v2, line="250A-BA-001")
    b.pipe(v2, v3, line="250A-FO-001")
    c = b.obj(S.OFF_PAGE_CONNECTOR, 500, 100, name="TO BILGE MAIN")
    b.pipe(v3, c, line="250A-BG-001")
    out = findings(b.build(), "DQ_LINE_SYSTEM_CODE")
    assert [f.verdict for f in out] == [V.FAIL]
    assert out[0].object_ids == ["p2"] and "interface" in out[0].description
    assert verdicts(DocBuilder().build(), "DQ_LINE_SYSTEM_CODE") == [V.NOT_APPLICABLE]


# --------------------------------------------------------------------------- lists


def test_equipment_and_valve_lists():
    b, ids = pump_station(ST.BALLAST, source_name="NO.1 W.B.TK (P)")
    doc = b.build()
    eq = {r["id"]: r for r in equipment_list(doc)}
    p0 = eq[ids["pumps"][0]]
    assert p0["tag"] == "P-01" and p0["name"] == "NO.1 PUMP" and p0["type_ko"] == "원심 펌프"
    assert p0["connections"] == 2 and p0["isolation_valves"] == "V-101; V-103"
    vl = {r["id"]: r for r in valve_list(doc)}
    cv = vl[ids["nrvs"][0]]
    assert cv["class"] == "check_valve" and cv["actuator"] == "none" and cv["remote_operated"] is False
    assert "P-01" in cv["connected_to"]


def test_valve_list_size_line_and_medium():
    doc, ids = _fo_transfer()
    row = next(r for r in valve_list(doc) if r["id"] == ids["qv"])
    assert row["size"] == "100A" and row["line_numbers"] == "100A-FO-001" and row["medium"] == "fuel_oil"
    assert row["actuator"] == "quick_closing" and row["actuator_ko"] == "급속차단" and row["remote_operated"]


@pytest.mark.parametrize("letters,variable,funcs", [
    ("PIC", "pressure", ["indicator", "controller"]),
    ("TCV", "temperature", ["controller", "valve"]),
    ("LAL", "level", ["alarm low"]),
    ("LAHH", "level", ["alarm high high"]),
    ("FQI", "flow integrate/totalize", ["indicator"]),
    ("PDI", "pressure differential", ["indicator"]),
    ("QIA", "quality (e.g. oil content ppm)", ["indicator", "alarm"]),
    ("LSH", "level", ["switch high"]),
    ("LSL", "level", ["switch low"]),
    ("PSLL", "pressure", ["switch low low"]),
    ("LSHH", "level", ["switch high high"]),
    ("TSHH", "temperature", ["switch high high"]),
    ("LIAHL", "level", ["indicator", "alarm high low"]),
    ("ZSO", "position", ["switch open"]),
    ("ZSC", "position", ["switch closed"]),
    ("HS", "hand", ["switch"]),
    ("XA", "unclassified", ["alarm"]),
    ("EI", "voltage", ["indicator"]),
    ("WT", "weight/force", ["transmitter"]),
    ("PSV", "pressure safety", ["valve"]),
    ("TSE", "temperature safety", ["element"]),
])
def test_decode_isa(letters, variable, funcs):
    assert decode_isa(letters) == (variable, funcs)


def test_instrument_list_letters_from_tag_text_and_attributes():
    b = DocBuilder()
    v = b.obj(S.GLOBE_VALVE, 300, 100, tag="PCV-101", actuator=A.DIAPHRAGM)
    j = b.junction(200, 100)
    b.chain(b.obj(S.GATE_VALVE, 100, 100), j, v)
    i1 = b.obj(S.INSTRUMENT_PANEL, 200, 40, tag="PIC 101")
    i2 = b.obj(S.INSTRUMENT_LOCAL, 400, 40)
    b.text("TI", 395, 33, role=TextRole.INSTRUMENT, owner=i2)
    b.text("205", 395, 41, role=TextRole.INSTRUMENT, owner=i2)
    i3 = b.obj(S.INSTRUMENT_SHARED, 500, 40, tag="X-1", attrs={"letters": "LT", "loop": "301"})
    b.pipe(i1, j, kind=PipeKind.INSTRUMENT_TAP)
    b.pipe(i1, v, kind=PipeKind.SIGNAL)
    b.pipe(i2, v, kind=PipeKind.INSTRUMENT_TAP)
    rows = {r["id"]: r for r in instrument_list(b.build())}
    assert rows[i1]["letters"] == "PIC" and rows[i1]["loop"] == "101"
    assert rows[i1]["description"] == "pressure indicator controller"
    assert rows[i1]["location"] == "panel / remote" and rows[i1]["signal_to"] == "PCV-101"
    assert rows[i2]["letters"] == "TI" and rows[i2]["loop"] == "205" and rows[i2]["measured_on"] == "PCV-101"
    assert rows[i3]["letters"] == "LT" and rows[i3]["loop"] == "301" and rows[i3]["function"] == "transmitter"


def test_line_list_groups_runs_through_valves():
    b = DocBuilder(ST.BALLAST)
    t1 = b.obj(S.TANK, 60, 100, name="NO.1 W.B.TK (P)")
    v1, v2 = b.obj(S.GATE_VALVE, 160, 100, tag="BA-V001"), b.obj(S.GATE_VALVE, 260, 100, tag="BA-V002")
    t2 = b.obj(S.TANK, 400, 100, name="NO.2 W.B.TK (P)")
    b.pipe(t1, v1, line="250A-BA-001-1S")
    b.pipe(v1, v2)  # unnumbered run continues the line
    b.pipe(v2, t2, line="250A-BA-001-1S")
    t3 = b.obj(S.TANK, 60, 300, name="NO.3 W.B.TK (P)")
    b.pipe(t3, b.obj(S.GATE_VALVE, 160, 300), line="150A-BA-002")
    rows = line_list(b.build())
    assert [r["line_number"] for r in rows] == ["150A-BA-002", "250A-BA-001-1S"]
    r = rows[1]
    assert r["pipe_ids"] == "p1 p2 p3" and r["runs"] == 3 and r["size"] == "250A" and r["system_code"] == "BA"
    assert r["spec"] == "1S" and r["medium"] == "ballast_water"
    assert r["from"] == "NO.1 W.B.TK (P)" and r["to"] == "NO.2 W.B.TK (P)"
    assert r["valves_fittings"] == "BA-V001; BA-V002"
    assert r["length_pt"] == pytest.approx(sum(p.length for p in b.pipes[:3]), abs=0.2)


def test_line_list_never_merges_two_numbers():
    b = DocBuilder()
    v1, v2, v3 = b.obj(S.GATE_VALVE, 100, 100), b.obj(S.GATE_VALVE, 200, 100), b.obj(S.GATE_VALVE, 300, 100)
    b.pipe(v1, v2, line="100A-BA-001")
    b.pipe(v2, v3, line="100A-BA-002")
    assert len(line_list(b.build())) == 2


# --------------------------------------------------------------------------- analyzer


def test_analyze_fills_report_deterministically():
    doc, ids = _bilge()
    rep = analyze(doc)
    assert isinstance(rep, AnalysisReport) and rep.document is doc
    assert rep.counts["objects"] == len(doc.objects) and rep.counts["valves"] > 5
    assert rep.counts["findings"] == len(rep.findings) > 0
    assert rep.counts["findings_fail"] + rep.counts["findings_pass"] + rep.counts["findings_insufficient_data"] + \
        rep.counts["findings_not_applicable"] == len(rep.findings)
    assert rep.equipment_list and rep.valve_list and rep.instrument_list and rep.line_list
    assert rep.flow_paths and all(f.source == "rule" for f in rep.findings)
    assert rep.meta["summary_source"] == "template" and rep.meta["review_status"] == "none"
    assert rep.meta["systems"] == ["bilge"] and rep.meta["system_basis"] == "drawing_info"
    assert rep.summary_en.startswith("[Deterministic summary") and "no language model" in rep.summary_en
    assert rep.summary_ko.startswith("[결정론적 요약") and "빌지 계통" in rep.summary_ko
    assert rep.llm.backend == "none" and rep.llm.calls == 0
    again = analyze(doc)
    assert again.model_dump() == rep.model_dump()


def test_analyze_summary_lists_failures():
    doc, _ = _bilge(nrv_on_well2=False)
    rep = analyze(doc)
    assert "[R_BILGE_SUCTION_NRV]" in rep.summary_en and "[R_BILGE_SUCTION_NRV]" in rep.summary_ko
    assert "빌지 흡입 분기관" in rep.summary_ko


def test_sorted_findings_order():
    doc, _ = _bilge(nrv_on_well2=False)
    order = [f.verdict for f in sorted_findings(analyze(doc).findings)]
    rank = {V.FAIL: 0, V.INSUFFICIENT_DATA: 1, V.PASS: 2, V.NOT_APPLICABLE: 3}
    assert order == sorted(order, key=rank.get) and order[0] == V.FAIL


class _AddingReviewer:
    def review(self, report: AnalysisReport) -> AnalysisReport:
        report.findings.append(Finding(id="mock-001", rule_id="MOCK-EXTRA", verdict=V.FAIL, severity=Severity.WARNING,
                                       title="extra", description="from reviewer", source="mock"))
        report.meta["summary_source"] = "mock"
        return report


class _FailingReviewer:
    def review(self, report: AnalysisReport) -> AnalysisReport:
        report.findings.clear()  # mutates its copy, then fails
        raise TimeoutError("gemma endpoint timed out")


class _WrongTypeReviewer:
    def review(self, report: AnalysisReport):
        return {"not": "a report"}


class _SilentlyFailingReviewer:
    """Handles its own failure: returns the copy unchanged with an error appended, no status."""

    def review(self, report: AnalysisReport) -> AnalysisReport:
        report.llm.errors.append("review: upstream 503")
        return report


def _gemma_review(responder) -> AnalysisReport:
    from shippid.analysis.gemma import GemmaAnalyzer
    from shippid.llm.base import LLMClient, MockBackend

    doc, _ = _bilge()
    return analyze(doc, reviewer=GemmaAnalyzer(LLMClient(MockBackend(responder), max_attempts=1)))


def test_review_status_reports_gemma_analyzer_failure_as_error(tmp_path):
    from shippid.llm.base import LLMError
    from shippid.report.html import write_html_report

    def refuse(msgs, schema, name):
        raise LLMError("connection refused: http://localhost:8000/v1/chat/completions")

    rep = _gemma_review(refuse)
    assert rep.meta["review"]["status"] == "error"
    assert rep.meta["review_status"] == "error" and rep.meta["reviewer"] == "GemmaAnalyzer"
    assert rep.meta["summary_source"] == "template" and rep.summary_en.startswith("[Deterministic summary")
    assert any("connection refused" in e for e in rep.llm.errors)
    html = write_html_report(rep, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<dt>검토 상태 Review</dt><dd>error</dd>" in html
    assert "<dt>요약 출처 Summary source</dt><dd>template</dd>" in html


def test_review_status_ok_for_successful_gemma_analyzer_review():
    def answer(msgs, schema, name):
        return {"findings": [], "summary_en": "Mock review summary.", "summary_ko": "모의 검토 요약입니다."}

    rep = _gemma_review(answer)
    assert rep.meta["review"]["status"] == "ok" and rep.meta["review_status"] == "ok"
    assert rep.meta["summary_source"] == "mock" and rep.summary_ko == "모의 검토 요약입니다."


def test_review_status_from_errors_when_reviewer_records_no_status():
    doc, _ = _bilge()
    rep = analyze(doc, reviewer=_SilentlyFailingReviewer())
    assert rep.meta["review_status"] == "error" and rep.llm.errors == ["review: upstream 503"]
    base = analyze(doc)
    assert review_status(base) == "ok"
    base.llm.errors.append("refine: earlier stage error")
    assert review_status(base, n_errors_before=1) == "ok"  # errors from before the review do not count
    base.meta["review"] = {"status": "ERROR"}
    assert review_status(base, n_errors_before=1) == "error"


def test_reviewer_success_failure_and_wrong_type():
    doc, _ = _bilge()
    base = analyze(doc)
    assert isinstance(_AddingReviewer(), ReportReviewer)
    rep = analyze(doc, reviewer=_AddingReviewer())
    assert rep.meta["review_status"] == "ok" and rep.meta["reviewer"] == "_AddingReviewer"
    assert len(rep.findings) == len(base.findings) + 1 and rep.findings[-1].source == "mock"
    assert rep.meta["summary_source"] == "mock"

    rep = analyze(doc, reviewer=_FailingReviewer())
    assert rep.meta["review_status"] == "error"
    assert len(rep.findings) == len(base.findings)  # deterministic report kept intact
    assert any("TimeoutError" in e and "gemma endpoint timed out" in e for e in rep.llm.errors)
    assert rep.meta["summary_source"] == "template"

    rep = analyze(doc, reviewer=_WrongTypeReviewer())
    assert rep.meta["review_status"] == "error" and "expected AnalysisReport" in rep.llm.errors[0]


def test_analyze_handles_degenerate_documents():
    rep = analyze(PIDDocument())
    assert rep.counts["objects"] == 0 and rep.flow_paths == [] and rep.findings
    b = DocBuilder()
    b.unknown(100, 100)
    b.pipe(None, None, a_xy=(0, 0), b_xy=(50, 0))
    rep = analyze(b.build())
    assert any(f.rule_id == "DQ_DANGLING_PIPE_ENDS" and f.verdict == V.FAIL for f in rep.findings)


def test_analyze_records_truncated_flow_paths():
    doc, _ = _sea_suction(2, True, ST.SEA_WATER_COOLING)
    full = analyze(doc)
    rep = analyze(doc, max_paths=1)
    assert len(rep.flow_paths) == 1 and rep.meta["flow_paths_truncated"]
    assert rep.meta["flow_paths_total"] == full.meta["flow_paths_total"] == len(full.flow_paths) == 4


def test_flow_net_search_blocked_nodes():
    doc, ids = _sea_suction(1, True, ST.SEA_WATER_COOLING, n_pumps=1)
    net = FlowNet(doc)
    strainer = next(o.id for o in doc.objects if o.cls == S.STRAINER)
    pump = ids["pumps"][0]
    assert net.search(ids["chests"][0]).reached(pump)
    assert not net.search(ids["chests"][0], blocked=[strainer]).reached(pump)
    res = net.search(ids["chests"][0], stop=[strainer])  # reached, but not passed through
    assert res.reached(strainer) and not res.reached(pump)


def test_count_rules_on_secondary_system_without_equipment():
    b, _ = pump_station(ST.FUEL_OIL_TRANSFER)
    doc = b.build()
    doc.info.systems = [ST.FIRE_MAIN]  # e.g. picked up from a note mentioning the fire control station
    assert verdicts(doc, "R_FIRE_PUMPS_MIN2") == [V.NOT_APPLICABLE]
    assert verdicts(doc, "R_FIRE_PUMP_DISCHARGE_NRV") == [V.NOT_APPLICABLE]  # "NO.n PUMP" are not fire pumps here
    doc.info.system, doc.info.systems = ST.FIRE_MAIN, []
    assert verdicts(doc, "R_FIRE_PUMPS_MIN2") == [V.PASS]  # on a fire main drawing they are
    b = DocBuilder(ST.FIRE_MAIN)
    b.pipe(b.obj(S.GATE_VALVE, 100, 100), b.obj(S.OFF_PAGE_CONNECTOR, 200, 100, name="TO HYDRANTS"))
    assert verdicts(b.build(), "R_FIRE_PUMPS_MIN2") == [V.INSUFFICIENT_DATA]


def test_tank_capacity_parsing_for_quick_closing_rule():
    from shippid.analysis.rules import _capacity_litres

    assert _capacity_litres("NO.2 L.O. STOR. TK") is None  # "2 L" of "NO.2 L.O." is not a capacity
    assert _capacity_litres("H.F.O. SETT. TK 2,600 M3") == 2_600_000
    assert _capacity_litres("M.D.O. DAY TK 0.4 M3") == pytest.approx(400)
    assert _capacity_litres("F.O. TK 450L") == 450
    b = DocBuilder(ST.LUBE_OIL)
    small = b.obj(S.TANK, 60, 100, name="M.D.O. DAY TK 0.4 M3")
    big = b.obj(S.TANK, 60, 300, name="NO.2 L.O. STOR. TK")
    for t, y in ((small, 100), (big, 300)):
        b.chain(t, b.obj(S.GLOBE_VALVE, 160, y, actuator=A.QUICK_CLOSING), b.obj(S.PUMP_SCREW, 260, y))
    out = findings(b.build(), "R_QC_VALVE_FO_LO_TANK")
    assert [f.object_ids[0] for f in out] == [big] and out[0].verdict == V.PASS  # < 500 L tanks are exempt


# --------------------------------------------------------------------------- pages not analysed (scanned / empty)


def _scanned_document(**meta) -> PIDDocument:
    from shippid.model import PageInfo

    return PIDDocument(source="scanned.pdf", pages=[PageInfo(index=0, width=1191, height=842)],
                       meta={"unreadable_pages": [0], **meta})


def test_scanned_document_is_reported_not_passed():
    """An image-only page yields no objects: no rule may pass or be 'not applicable', and the
    summaries name the page that was not analysed (it used to read 'No rule check failed')."""
    rep = analyze(_scanned_document())
    assert rep.meta["unreadable_pages"] == [0] and rep.meta["empty_pages"] == []
    assert {f.rule_id for f in rep.findings} == set(RULES)
    assert all(f.verdict == V.INSUFFICIENT_DATA for f in rep.findings)
    assert all("Page(s) 1 scanned / raster-only" in f.description for f in rep.findings)
    assert rep.counts["findings_pass"] == 0 == rep.counts["findings_not_applicable"]
    en, ko = rep.summary_en, rep.summary_ko
    assert "WARNING: Page(s) 1 scanned / raster-only (unreadable) - these pages were not analysed." in en
    assert "No rule check failed" not in en and "nothing was extracted" in en
    assert "1페이지는 스캔/래스터 이미지라 판독 불가" in ko and "부적합 항목이 없습니다" not in ko
    # a reviewer-less report keeps the deterministic banner first
    assert en.startswith("[Deterministic summary")


def test_partly_scanned_document_keeps_findings_and_warns():
    b, _ = pump_station(ST.BALLAST)
    clean = b.build()
    doc = clean.model_copy(deep=True)
    doc.meta.update({"unreadable_pages": [1], "empty_pages": [2, "x", -1, True]})  # invalid entries ignored
    rep, ref = analyze(doc), analyze(clean)
    assert [(f.rule_id, f.verdict) for f in rep.findings] == [(f.rule_id, f.verdict) for f in ref.findings]
    assert rep.meta["unreadable_pages"] == [1] and rep.meta["empty_pages"] == [2]
    assert "page(s) 2 scanned / raster-only" in rep.summary_en.lower() and "page(s) 3 empty" in rep.summary_en
    assert "readable pages only" in rep.summary_en and "판독 가능한 페이지만" in rep.summary_ko
    assert "WARNING" not in ref.summary_en and ref.meta["unreadable_pages"] == []


def test_quality_checks_never_pass_vacuously():
    empty = DocBuilder(ST.BALLAST).build()
    for rid in ("DQ_UNCONNECTED_OBJECTS", "DQ_OPEN_INLINE_ITEMS", "DQ_DANGLING_PIPE_ENDS", "DQ_DUPLICATE_TAGS",
                "DQ_MISSING_TAGS"):
        assert verdicts(empty, rid) == [V.INSUFFICIENT_DATA], rid
    # objects but no pipes, no in-line items, no tags, nothing taggable
    b = DocBuilder(ST.BALLAST)
    c1, c2 = b.obj(S.OFF_PAGE_CONNECTOR, 100, 100), b.obj(S.OFF_PAGE_CONNECTOR, 300, 100)
    doc = b.build()
    assert verdicts(doc, "DQ_DANGLING_PIPE_ENDS") == [V.INSUFFICIENT_DATA]
    assert verdicts(doc, "DQ_OPEN_INLINE_ITEMS") == [V.NOT_APPLICABLE]
    assert verdicts(doc, "DQ_DUPLICATE_TAGS") == [V.NOT_APPLICABLE]
    assert verdicts(doc, "DQ_MISSING_TAGS") == [V.NOT_APPLICABLE]
    b.pipe(c1, c2)
    out = findings(b.build(), "DQ_DANGLING_PIPE_ENDS")
    assert out[0].verdict == V.PASS and "All 1 pipe runs" in out[0].description


# --------------------------------------------------------------------------- bounded flow-path enumeration


def _manifold(n_src: int, n_pumps: int, n_dst: int) -> PIDDocument:
    """Tanks -> suction header -> parallel pumps -> discharge header -> tanks: n_src * n_pumps * n_dst
    driven paths."""
    b = DocBuilder(ST.BALLAST)
    width = max(n_src, n_pumps, n_dst)
    suc = [b.junction(100 + 40 * i, 200) for i in range(width)]
    dis = [b.junction(100 + 40 * i, 400) for i in range(width)]
    b.chain(*suc)
    b.chain(*dis)
    for i in range(n_src):
        b.pipe(b.obj(S.TANK, 100 + 40 * i, 100, name=f"NO.{i + 1} W.B.TK"), suc[i], flow=1)
    for i in range(n_dst):
        b.pipe(dis[i], b.obj(S.TANK, 100 + 40 * i, 500, name=f"NO.{i + 1} F.W.TK"), flow=1)
    for i in range(n_pumps):
        b.chain(suc[i], b.obj(S.PUMP_CENTRIFUGAL, 100 + 40 * i, 300, name=f"NO.{i + 1} BALLAST PUMP"), dis[i])
    return b.build()


def test_flow_path_enumeration_is_bounded_and_exact():
    import tracemalloc

    from shippid.analysis.graph import enumerate_flow_paths

    doc = _manifold(12, 3, 12)
    G = build_graph(doc)
    full = enumerate_flow_paths(doc, G, 10 ** 6, max_work=None)
    assert full.total == len(full.paths) == 12 * 3 * 12 and full.exact and full.complete
    order = {n: G.nodes[n]["order"] for n in G.nodes}
    keys = [(order[f.source_id], order[f.target_id], len(f.node_ids), tuple(f.node_ids)) for f in full.paths]
    assert keys == sorted(keys)  # source, sink, length, nodes
    for k in (0, 1, 7, 40, 431, 432, 1000):
        res = enumerate_flow_paths(doc, G, k)
        assert res.paths == full.paths[:k] and res.total == full.total and res.exact
    assert flow_paths(doc, G, 5) == full.paths[:5]
    # out of budget: the count is a lower bound, the kept (first) paths are still the right ones
    capped = enumerate_flow_paths(doc, G, 7, max_work=150)
    assert not capped.exact and capped.complete and 0 < capped.total < full.total and capped.work == 150
    assert capped.paths == full.paths[:7]
    starved = enumerate_flow_paths(doc, G, 50, max_work=10)
    assert not starved.exact and not starved.complete and len(starved.paths) < 50
    # memory: only the kept paths are stored, not every path (the former code stored all of them)
    big = _manifold(30, 6, 30)  # 5400 paths
    Gb = build_graph(big)
    tracemalloc.start()
    res = enumerate_flow_paths(big, Gb, 5)
    _, peak_bounded = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    enumerate_flow_paths(big, Gb, 10 ** 6, max_work=None)
    _, peak_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert res.total == 5400 and len(res.paths) == 5 and peak_bounded < peak_full / 4


def test_analyze_reports_flow_path_budget():
    doc = _manifold(12, 3, 12)
    rep = analyze(doc, max_paths=5)
    assert rep.meta["flow_paths_total"] == 432 and rep.meta["flow_paths_total_exact"] and len(rep.flow_paths) == 5
    assert rep.meta["flow_paths_truncated"] and "Flow paths: 432 driven paths" in rep.summary_en
    rep = analyze(doc, max_paths=5, max_path_work=150)
    assert not rep.meta["flow_paths_total_exact"] and rep.meta["flow_paths_truncated"]
    assert rep.meta["flow_paths_complete"] and rep.meta["flow_paths_work"] == 150
    assert f"Flow paths: at least {rep.meta['flow_paths_total']} driven paths" in rep.summary_en
    assert f"펌프 구동 경로 최소 {rep.meta['flow_paths_total']}개" in rep.summary_ko


# --------------------------------------------------------------------------- analysis cost (no per-row / per-rule rescans)


def test_rule_context_caches_object_and_class_indexes(monkeypatch):
    from shippid.analysis.graph import PUMP_CLASSES, object_ids
    from shippid.analysis.rules import RuleContext

    b, ids = pump_station(ST.BALLAST, n_pumps=3)
    doc = b.build()
    ctx = RuleContext(doc)
    pumps = ctx.ids(*PUMP_CLASSES)
    assert pumps == [n for n in object_ids(ctx.G) if ctx.eff(n) in PUMP_CLASSES] == ids["pumps"]
    pumps.append("junk")  # callers get copies
    ctx.object_ids().clear()
    assert ctx.ids(*PUMP_CLASSES) == ids["pumps"] and ctx.object_ids() == object_ids(ctx.G)
    for p in ids["pumps"]:
        for n in ctx.G.nodes:
            assert ctx.on_other_pump_branch(p, n) == any(n in ctx.passed(q) for q in ids["pumps"] if q != p)
    calls = []
    real = rules_mod.object_ids
    monkeypatch.setattr(rules_mod, "object_ids", lambda G: calls.append(1) or real(G))
    run_rules(doc)
    assert len(calls) == 1  # was once per ids() call (hundreds on a large set)


def test_lists_build_the_object_index_once(monkeypatch):
    b, _ = pump_station(ST.BALLAST, n_pumps=3)
    i = b.obj(S.INSTRUMENT_LOCAL, 520, 230)
    b.text("PI", 520, 226, role=TextRole.INSTRUMENT, owner=i)
    b.text("101", 520, 234, role=TextRole.INSTRUMENT, owner=i)
    doc = b.build()
    G = build_graph(doc)
    expected = (equipment_list(doc, G), valve_list(doc, G), instrument_list(doc, G))
    calls = []
    real = PIDDocument.object_by_id
    monkeypatch.setattr(PIDDocument, "object_by_id", lambda self: calls.append(1) or real(self))
    got = (equipment_list(doc, G), valve_list(doc, G), instrument_list(doc, G))
    assert got == expected and len(calls) == 3  # one per list, not one per row
    assert [(r["letters"], r["loop"]) for r in got[2] if r["id"] == i] == [("PI", "101")]
