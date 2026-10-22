"""End-to-end tests of the full pipeline (PDF -> extraction -> analysis -> 3D -> reports).

The Gemma path is exercised with a scripted MockBackend: it returns schema-valid but
deliberately uninformative answers, so the test checks the plumbing (calls made, usage
recorded, honest fallbacks) -- not model quality, which needs a real Gemma 4 31B server.
"""

import json
import re
import struct

import pytest

from shippid.cli import main as cli_main
from shippid.config import Config
from shippid.datasets.pdf_writer import write_pdf
from shippid.datasets.synthetic import generate
from shippid.llm.base import LLMClient, MockBackend
from shippid.model import SystemType
from shippid.pipeline import run_pipeline, summarize


@pytest.fixture(scope="module")
def ballast_pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("pdf")
    gt, spec = generate(11, SystemType.BALLAST)
    path = write_pdf(spec, d / "ballast.pdf")
    gt.save(d / "ballast.json")
    return path, gt


def _glb_json(path):
    data = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF"
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A  # JSON
    return json.loads(data[20:20 + chunk_len])


def test_pipeline_deterministic(ballast_pdf, tmp_path):
    pdf, gt = ballast_pdf
    cfg = Config()
    cfg.llm.backend = "none"
    res = run_pipeline(pdf, tmp_path / "out", cfg, viewer="both")
    s = summarize(res)
    assert s["llm"] == "none"
    assert res.document.info.system == SystemType.BALLAST
    assert len(res.document.objects) >= 0.9 * len(gt.objects)
    for key in ("extracted", "document", "analysis", "layout", "glb", "viewer:threejs", "viewer:plotly", "report"):
        assert res.paths[key].is_file(), key
    assert res.layout.planner == "heuristic"
    assert {p.object_id for p in res.layout.placements} == {o.id for o in res.document.objects}
    gltf = _glb_json(res.paths["glb"])
    names = [n.get("name", "") for n in gltf["nodes"]]
    assert len(names) == len(set(names))
    assert not any(re.search(r"[\s\[\].:/]", n) for n in names)
    html = res.paths["report"].read_text(encoding="utf-8")
    assert "viewer.html" in html and "R_" in html
    assert res.report.llm.calls == 0


def _scripted_gemma(messages, schema, name):
    """Schema-valid, uninformative answers (never adopted over the vector result)."""
    text = "\n".join(m.text() for m in messages if m.role == "user")
    if name == "SymbolBatch":
        ids = list(dict.fromkeys(re.findall(r'"id":\s*"(o\d+)"', text)))
        return {"items": [{"id": i, "reason": "scripted", "cls": "unknown", "actuator": "none", "confidence": 0.0}
                          for i in ids]}
    if name == "DrawingInfoOut":
        return {"system": "ballast", "systems": ["ballast"]}
    if name == "TextAssignments":
        return {"assignments": []}
    if name == "ReviewOut":
        return {"findings": [], "summary_en": "scripted review", "summary_ko": "스크립트 검토 결과"}
    return {"items": []}  # layout plan: empty -> the planner must fall back honestly


def test_pipeline_with_llm_client(ballast_pdf, tmp_path):
    pdf, _ = ballast_pdf
    backend = MockBackend(_scripted_gemma, model="scripted-gemma")
    client = LLMClient(backend, max_attempts=2)
    res = run_pipeline(pdf, tmp_path / "out", Config(), client=client)
    assert backend.calls, "the Gemma steps must call the LLM"
    called = {name for _, _, name in backend.calls}
    assert {"ReviewOut"} <= called
    assert res.report.llm.calls > 0
    assert res.report.summary_ko == "스크립트 검토 결과"
    # an empty layout plan is not a Gemma plan: the arrangement must be labelled heuristic
    assert res.layout.planner == "heuristic"
    assert {p.object_id for p in res.layout.placements} == {o.id for o in res.document.objects}
    # uninformative symbol answers never override the vector classification
    vec = json.loads((tmp_path / "out" / "extracted.json").read_text())
    assert [o["cls"] for o in vec["objects"]] == [o.cls.value for o in res.document.objects]


def test_cli_extract_and_symbols(ballast_pdf, tmp_path, capsys):
    pdf, _ = ballast_pdf
    out = tmp_path / "x.json"
    assert cli_main(["extract", str(pdf), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["objects"]
    legend = tmp_path / "legend.pdf"
    assert cli_main(["symbols", "-o", str(legend)]) == 0
    assert legend.stat().st_size > 1000


def test_cli_llm_check_without_backend(capsys, monkeypatch):
    monkeypatch.delenv("SHIPPID_LLM_BACKEND", raising=False)
    assert cli_main(["llm-check", "--llm", "none"]) == 2


# --------------------------------------------------------------------------- multi-sheet sets, benchmark, CLI errors

from shippid.benchmark import prediction_name, run_benchmark  # noqa: E402
from shippid.datasets.multisheet import generate_multisheet  # noqa: E402
from shippid.model import AnalysisReport, PIDDocument  # noqa: E402


@pytest.fixture(scope="module")
def multisheet_set(tmp_path_factory):
    """A 2-sheet ballast set (one continuation pair) as PDF + ground truth + benchmark manifest."""
    d = tmp_path_factory.mktemp("ms")
    gt, pdf = generate_multisheet(4, SystemType.BALLAST, sheets=2, style="mark")
    assert gt.meta["sheet_links"]
    (d / "ms4.pdf").write_bytes(pdf)
    gt.save(d / "ms4.json")
    (d / "manifest.jsonl").write_text(json.dumps({"id": "ms4", "pdf": "ms4.pdf", "gt": "ms4.json", "split": "test"}) + "\n")
    return d, gt


def _none_cfg(**kw):
    cfg = Config()
    cfg.llm.backend = "none"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_pipeline_multisheet_artefacts_carry_the_sheet_links(multisheet_set, tmp_path):
    d, gt = multisheet_set
    res = run_pipeline(d / "ms4.pdf", tmp_path / "out", _none_cfg(), stages=("extract", "refine", "analyze"))
    doc = PIDDocument.load(res.paths["document"])
    extracted = PIDDocument.load(res.paths["extracted"])
    report = AnalysisReport.model_validate_json(res.paths["analysis"].read_text(encoding="utf-8"))
    for key in ("sheet_links", "sheet_links_ambiguous", "sheet_link_stats"):
        assert key in doc.meta and key in extracted.meta, key
    assert doc.meta["sheet_links"] and doc.meta["sheet_links"] == report.document.meta["sheet_links"]
    assert doc.model_dump() == report.document.model_dump()  # document.json is the analysed document
    assert "link_sheets" in res.timings
    out = tmp_path / "x.json"  # `shippid extract` links the sheets as `run` does
    assert cli_main(["extract", str(d / "ms4.pdf"), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["meta"]["sheet_links"] == doc.meta["sheet_links"]


def test_benchmark_pairs_sheets_and_scores_cross_sheet_connections(multisheet_set, tmp_path):
    d, gt = multisheet_set
    import shippid.evaluate as ev

    calls = []
    orig = ev.evaluate_document
    try:
        ev.evaluate_document = lambda *a, **k: calls.append(1) or orig(*a, **k)
        res = run_benchmark(d, tmp_path / "bench", workers=1, cfg=_none_cfg(), save_predictions=True)
    finally:
        ev.evaluate_document = orig
    assert len(calls) == 1  # every document is evaluated once
    conn = res["micro"]["connections"]
    assert conn["fn"] == 0 and conn["fp"] == 0, conn  # the continuation pair is a ground-truth connection
    pred = PIDDocument.load(tmp_path / "bench" / "predictions" / "ms4.pred.json")
    assert pred.meta["sheet_links"]
    row = res["benchmark"]["per_document"][0]
    assert row["id"] == "ms4" and row["connection_f1"] == pytest.approx(1.0) and row["seconds"] > 0
    assert res["benchmark"]["llm"] == "none"


def test_benchmark_uses_the_given_config_in_the_workers(multisheet_set, tmp_path):
    from shippid.extract.pipeline import extract_pdf

    d, _ = multisheet_set
    cfg = _none_cfg(snap_tol=30.0)  # a tolerance that visibly changes this drawing's pipe network
    direct = extract_pdf(d / "ms4.pdf", cfg)
    default = extract_pdf(d / "ms4.pdf", _none_cfg())
    assert (len(direct.objects), len(direct.pipes)) != (len(default.objects), len(default.pipes))
    run_benchmark(d, tmp_path / "b", workers=2, cfg=cfg, save_predictions=True)
    pred = PIDDocument.load(tmp_path / "b" / "predictions" / "ms4.pred.json")
    assert (len(pred.objects), len(pred.pipes)) == (len(direct.objects), len(direct.pipes))


def test_benchmark_prediction_files_stay_inside_the_output(ballast_pdf, tmp_path):
    pdf, _ = ballast_pdf
    gt_path = pdf.with_suffix(".json")
    data = tmp_path / "data"
    data.mkdir()
    ids = ["../../ESCAPED", str(tmp_path / "abs_target"), "a b/c", "X", "x", ".."]
    (data / "manifest.jsonl").write_text("".join(
        json.dumps({"id": i, "pdf": str(pdf), "gt": str(gt_path), "split": "test"}) + "\n" for i in ids))
    out = tmp_path / "work" / "out"
    res = run_benchmark(data, out, workers=1, cfg=_none_cfg(), save_predictions=True)
    assert res["benchmark"]["documents"] == len(ids)
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.pred.json"))
    assert all(w.startswith("work/out/predictions/") for w in written), written
    assert len(written) == len(ids)  # case-insensitive collisions (X / x) get distinct names
    assert prediction_name("../../ESCAPED", pdf) == "ESCAPED.pred.json"
    assert prediction_name("..", pdf) == f"{pdf.stem}.pred.json"


def test_benchmark_llm_label_and_concurrency(ballast_pdf, tmp_path, monkeypatch):
    from shippid.analysis import gemma

    pdf, _ = ballast_pdf
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.jsonl").write_text(json.dumps(
        {"id": "b", "pdf": str(pdf), "gt": str(pdf.with_suffix(".json")), "split": "test"}) + "\n")
    seen = []
    init = gemma.GemmaAnalyzer.__init__

    def spy(self, *a, **k):
        init(self, *a, **k)
        seen.append(self.concurrency)

    monkeypatch.setattr(gemma.GemmaAnalyzer, "__init__", spy)
    cfg = Config()
    cfg.llm.concurrency = 3
    client = LLMClient(MockBackend(_scripted_gemma, model="scripted-gemma"), max_attempts=2)
    res = run_benchmark(data, tmp_path / "bench", cfg=cfg, use_llm=True, client=client)
    bench = res["benchmark"]
    assert bench["llm"] == "mock:scripted-gemma" and bench["llm_usage"]["backend"] == "mock"
    assert "mock:scripted-gemma" in (tmp_path / "bench" / "evaluation.md").read_text(encoding="utf-8")
    assert seen == [3]
    # the pipeline wires LLMConfig.concurrency the same way
    run_pipeline(pdf, tmp_path / "run", cfg, client=client, stages=("extract", "refine"))
    assert seen == [3, 3]


def test_planner_gemma_without_llm_fails_before_any_stage(ballast_pdf, tmp_path, capsys):
    pdf, _ = ballast_pdf
    with pytest.raises(ValueError, match="planner='gemma' needs an LLM backend"):
        run_pipeline(pdf, tmp_path / "pg", _none_cfg(), planner="gemma")
    assert not (tmp_path / "pg").exists()  # nothing was extracted or written
    with pytest.raises(ValueError, match="planner must be one of"):
        run_pipeline(pdf, tmp_path / "pg", _none_cfg(), planner="magic")
    # analysis without the 3D stage does not need a planner
    run_pipeline(pdf, tmp_path / "an", _none_cfg(), planner="gemma", stages=("extract", "analyze"))
    out = tmp_path / "out"
    assert cli_main(["run", str(pdf), "-o", str(out), "--llm", "none", "--planner", "gemma"]) == 2
    assert not out.exists()
    run_pipeline(pdf, out, _none_cfg(), stages=("extract", "analyze"))
    capsys.readouterr()
    rc = cli_main(["model3d", str(out / "analysis.json"), "-o", str(tmp_path / "m3"), "--llm", "none",
                   "--planner", "gemma"])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 2 and len(err) == 1 and "planner='gemma' needs an LLM backend" in err[0]
    assert not (tmp_path / "m3").exists()


def test_cli_input_errors_are_one_line(ballast_pdf, tmp_path, capsys):
    import pymupdf

    pdf, _ = ballast_pdf
    enc = tmp_path / "encrypted.pdf"
    with pymupdf.open(str(pdf)) as d:
        d.save(str(enc), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    (tmp_path / "garbage.pdf").write_bytes(b"not a pdf " * 20)
    (tmp_path / "empty.pdf").write_bytes(b"")
    layout = tmp_path / "layout3d.json"
    layout.write_text(json.dumps({"schema_version": "1", "planner": "heuristic", "placements": [], "routes": []}))
    cases = {
        "encrypted": (["extract", str(enc)], "encrypted"),
        "garbage": (["extract", str(tmp_path / "garbage.pdf")], "cannot open PDF"),
        "empty": (["run", str(tmp_path / "empty.pdf"), "-o", str(tmp_path / "o1"), "--llm", "none"], "cannot open PDF"),
        "missing": (["analyze", str(tmp_path / "nope.pdf"), "-o", str(tmp_path / "o2"), "--llm", "none"], "nope.pdf"),
        "page": (["extract", str(pdf), "--pages", "9"], "page 9 out of range"),
        "layout": (["model3d", str(layout), "-o", str(tmp_path / "o3"), "--llm", "none"], "3D layout"),
        "evaluate": (["evaluate", "--data", str(tmp_path), "-o", str(tmp_path / "o4")], "manifest.jsonl"),
        "with_llm": (["evaluate", "--data", str(tmp_path), "-o", str(tmp_path / "o5"), "--with-llm", "--llm", "none"],
                     "--with-llm needs an LLM backend"),
    }
    for name, (argv, needle) in cases.items():
        capsys.readouterr()
        rc = cli_main(argv)
        err = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
        assert rc == 2, name
        assert len(err) == 1 and err[0].startswith("shippid: error: ") and needle in err[0], (name, err)
    # -v keeps the traceback for debugging
    capsys.readouterr()
    assert cli_main(["-v", "extract", str(pdf), "--pages", "9"]) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err and err.strip().splitlines()[-1].startswith("shippid: error: ")
    # malformed --pages is an argparse usage error
    with pytest.raises(SystemExit) as exc:
        cli_main(["extract", str(pdf), "--pages", "a,b"])
    assert exc.value.code == 2
    # an invalid PIDDocument JSON: one line naming the model and the first error
    bad = tmp_path / "bad_doc.json"
    bad.write_text(json.dumps({"objects": [{"id": "o1"}], "unexpected": 1}))
    capsys.readouterr()
    assert cli_main(["model3d", str(bad), "-o", str(tmp_path / "o6"), "--llm", "none"]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "invalid PIDDocument JSON" in err[0]
