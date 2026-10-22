"""Integration: extract PDFs written by the dataset generator and compare with its
ground truth using the shared evaluator (:func:`shippid.evaluate.evaluate_document`).
Skipped while :mod:`shippid.datasets` does not expose the API.

The extractor itself never imports :mod:`shippid.datasets`; only this test does.
"""

from __future__ import annotations

import dataclasses
import importlib
import random

import pytest

from shippid.model import SymbolClass


def _datasets():
    synthetic = pytest.importorskip("shippid.datasets.synthetic")
    if not hasattr(synthetic, "generate"):
        pytest.skip("shippid.datasets.synthetic.generate() not available yet")
    try:
        writer = importlib.import_module("shippid.datasets.pdf_writer")
    except ImportError:
        pytest.skip("shippid.datasets.pdf_writer not available yet")
    if not hasattr(writer, "write_pdf"):
        pytest.skip("shippid.datasets.pdf_writer.write_pdf() not available yet")
    return synthetic, writer


# (seed, style overrides): the generator's random styles plus forced variants that cover the
# sheet furniture and drawing conventions (symbol legend, OCG layers, ISO 14726 coloured pipes,
# gap crossings, one path per segment, interleaved drawing order, bottom title band, notes box)
CASES = [
    (1, {}),
    (2, {}),
    (3, {}),
    (4, {"legend": True, "layers": ("PIPE", "SYMBOL", "TEXT", "FRAME"), "color_pipes": True}),
    (5, {"crossing": "gap", "split_pipes": True, "interleave": True, "junction_dots": False}),
    (6, {"sheet_layout": "bottom", "notes_box": True, "legend": True, "flow_arrows": True}),
]


@pytest.mark.parametrize("seed,overrides", CASES, ids=[f"seed{s}" + ("-" + "-".join(o) if o else "") for s, o in CASES])
def test_generated_drawings(seed, overrides, tmp_path):
    synthetic, writer = _datasets()
    from shippid.evaluate import evaluate_document
    from shippid.extract.pipeline import extract_pdf

    style = None
    if overrides:
        style = dataclasses.replace(synthetic.random_style(random.Random(seed)), **overrides)
    try:
        gt, spec = synthetic.generate(seed, style=style)
    except Exception as exc:  # the generator may legitimately reject a layout
        if type(exc).__name__ == "LayoutError":
            pytest.skip(f"generator rejected seed {seed}: {exc}")
        raise
    path = writer.write_pdf(spec, tmp_path / f"gen_{seed}.pdf")
    doc = extract_pdf(path)
    assert all(o.source == "vector" for o in doc.objects)
    assert not [o for o in doc.objects if o.cls == SymbolClass.UNKNOWN]
    r = evaluate_document(doc, gt)
    m = {
        "det_recall": r["detection"]["recall"], "det_precision": r["detection"]["precision"],
        "class_acc": r["classification"]["accuracy"], "tag_acc": r["tag"]["accuracy"],
        "actuator_acc": r["actuator"]["accuracy"], "conn_precision": r["connections"]["precision"],
        "conn_recall": r["connections"]["recall"], "text_role_acc": r["texts"]["role"]["accuracy"],
        "text_owner_acc": r["texts"]["owner"]["accuracy"], "line_number_acc": r["line_number"]["accuracy"],
        "process_pipe_recall": r["pipes"]["process"]["recall"],
    }
    floors = {"det_recall": 0.97, "det_precision": 0.97, "class_acc": 0.97, "tag_acc": 0.97, "actuator_acc": 0.97,
              "conn_precision": 0.97, "conn_recall": 0.97, "text_role_acc": 0.97, "text_owner_acc": 0.95,
              "line_number_acc": 0.85, "process_pipe_recall": 0.97}
    low = {k: v for k, v in m.items() if v is not None and v < floors[k]}
    assert not low, (low, m)


# export variants (shippid.datasets.variants): seed 3 / 5 give PDFSHX=1 (annotations), seed 1 / 2 PDFSHX=2
VARIANT_CASES = [("shx_text", 1), ("shx_text", 3), ("fragmented_text", 1), ("polyline_curves", 2), ("yard_symbols", 2)]


@pytest.mark.parametrize("variant,seed", VARIANT_CASES, ids=[f"{v}-{s}" for v, s in VARIANT_CASES])
def test_export_variants(variant, seed, tmp_path):
    variants = pytest.importorskip("shippid.datasets.variants")
    from shippid.analysis.gemma import GemmaAnalyzer
    from shippid.evaluate import evaluate_document, match_boxes
    from shippid.extract.pipeline import ExtractOptions, extract_pdf
    from shippid.llm.base import LLMClient, MockBackend

    gt, pdf = variants.make_variant(seed, variant)
    path = tmp_path / f"{variant}_{seed}.pdf"
    path.write_bytes(pdf)
    # the yard_symbols calibration below is about symbol styles the extractor has not seen, i.e. without
    # learning them from the drawing's legend (legend learning: tests/test_legend.py)
    doc = extract_pdf(path, options=ExtractOptions(learn_legends=variant != "yard_symbols"))
    r = evaluate_document(doc, gt)
    assert r["detection"]["f1"] >= 0.99, r["detection"]
    assert r["connections"]["f1"] >= 0.99, r["connections"]
    assert r["texts"]["f1"] >= 0.99 and r["texts"]["content"]["accuracy"] >= 0.99, r["texts"]
    if variant == "shx_text":
        sources = doc.meta["pages"]["0"]["text_sources"]
        carrier = "annotation" if gt.meta["variant"]["params"]["shx_text"]["pdfshx"] == 1 else "hidden"
        assert sources[carrier] == len(gt.texts) and sources["text"] == 0
    if variant != "yard_symbols":
        assert r["detection_class_aware"]["f1"] >= 0.99
        return
    # unseen symbol styles: every misclassified object goes to review (Gemma), none with high confidence
    an = GemmaAnalyzer(LLMClient(MockBackend(lambda *a: {})))
    wrong = [doc.objects[i] for i, j, _ in match_boxes([o.bbox for o in doc.objects], [o.bbox for o in gt.objects], 0.5)
             if doc.objects[i].cls != gt.objects[j].cls]
    assert wrong and all(an.needs_review(o) and o.confidence < an.min_confidence for o in wrong)
