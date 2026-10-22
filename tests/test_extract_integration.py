"""Integration: extract PDFs written by the dataset generator and compare with its
ground truth. Skipped while :mod:`shippid.datasets` does not expose the API.

The extractor itself never imports :mod:`shippid.datasets`; only this test does.
"""

from __future__ import annotations

import importlib

import pytest

from shippid.model import BBox, PIDDocument


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


def _recall(gt: PIDDocument, pred: PIDDocument, iou: float = 0.5) -> tuple[float, float]:
    """Object recall and class accuracy over matched objects (greedy IoU matching)."""
    pairs = []
    for gi, g in enumerate(gt.objects):
        for pi, p in enumerate(pred.objects):
            if g.page == p.page:
                v = g.bbox.iou(p.bbox)
                if v >= iou:
                    pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, correct = set(), set(), 0
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        correct += gt.objects[gi].cls == pred.objects[pi].cls
    n = max(1, len(gt.objects))
    return len(used_g) / n, correct / max(1, len(used_g))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_generated_drawings_recall(seed, tmp_path):
    synthetic, writer = _datasets()
    from shippid.extract.pipeline import extract_pdf

    try:
        gt, spec = synthetic.generate(seed)
    except Exception as exc:  # the generator may legitimately reject a layout
        if type(exc).__name__ == "LayoutError":
            pytest.skip(f"generator rejected seed {seed}: {exc}")
        raise
    path = writer.write_pdf(spec, tmp_path / f"gen_{seed}.pdf")
    doc = extract_pdf(path)
    assert all(o.source == "vector" for o in doc.objects)
    recall, cls_acc = _recall(gt, doc)
    assert recall >= 0.9, f"object recall {recall:.3f} < 0.9 (class accuracy {cls_acc:.3f})"
    assert isinstance(gt.objects[0].bbox, BBox)
