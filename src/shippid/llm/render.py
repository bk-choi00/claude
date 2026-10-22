"""PyMuPDF rendering helpers that turn PDF regions into PNG images for Gemma 4.

Gemma 4's image processor keeps the aspect ratio and works on 48 px patches
(``max_soft_tokens`` 1120 is about 2.6 MP, e.g. 1680x1536 px), so every image
produced here has pixel dimensions that are multiples of :data:`PATCH` whenever
the size limit allows it. A region is never stretched: when its aspect ratio
does not fill the padded canvas exactly, the clip is widened symmetrically so
the extra pixels show real drawing context; only areas beyond the page edge
are white.

Only PyMuPDF is used (no PIL): labels on contact sheets are real text inserted
into a temporary PDF page which is then rasterised.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import pymupdf

PATCH = 48  # Gemma 4 vision patch size in pixels
DEFAULT_TILE_DPI = 144.0  # 7 pt tag text -> ~14 px high, legible for the model
_LABEL_FONT = "hebo"  # Helvetica-Bold (base-14, always available)


# --------------------------------------------------------------------------- helpers


def open_pdf(path: str | Path) -> pymupdf.Document:
    """Open a PDF for rendering with every optional-content layer switched on.

    Content on OCG layers that are OFF is silently skipped by MuPDF; the vector
    extractor turns all layers on, so crops must do the same to show what was
    extracted.
    """
    doc = pymupdf.open(str(path))
    if doc.needs_pass:
        doc.close()
        raise ValueError(f"{path}: encrypted PDF")
    try:
        ocgs = doc.get_ocgs()
    except Exception:  # pragma: no cover - non-PDF inputs
        ocgs = {}
    if ocgs and not all(v.get("on", True) for v in ocgs.values()):
        doc.set_layer(-1, basestate="ON", on=list(ocgs), off=[])
        data = doc.tobytes()
        doc.close()
        doc = pymupdf.open("pdf", data)
    return doc


def _floor_patch(n: float) -> int:
    return int(n // PATCH) * PATCH if n >= PATCH else max(1, int(n))


def _ceil_patch(n: float, limit: int) -> int:
    """Smallest multiple of PATCH >= n, capped at ``limit`` (which is patch-aligned when >= PATCH)."""
    v = int(math.ceil(n / PATCH - 1e-9)) * PATCH
    return max(1, min(max(v, PATCH if limit >= PATCH else 1), limit))


def _as_rect(rect) -> pymupdf.Rect:
    r = pymupdf.Rect(rect)
    if r.is_empty or r.is_infinite:
        r = pymupdf.Rect(r.x0, r.y0, max(r.x1, r.x0 + 1.0), max(r.y1, r.y0 + 1.0))
    return r


def _render_exact(src, center: pymupdf.Point, zoom: float, width: int, height: int) -> pymupdf.Pixmap:
    """RGB pixmap of exactly ``width`` x ``height`` px showing ``src`` (a page or its display
    list) around ``center`` at ``zoom``."""
    cw, ch = width / zoom, height / zoom
    clip = pymupdf.Rect(center.x - cw / 2, center.y - ch / 2, center.x + cw / 2, center.y + ch / 2)
    canvas = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    canvas.clear_with(255)
    visible = clip & src.rect
    if visible.is_empty or visible.width <= 0 or visible.height <= 0:
        return canvas
    pix = src.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=visible, alpha=False, colorspace=pymupdf.csRGB)
    pix.set_origin(int(round((visible.x0 - clip.x0) * zoom)), int(round((visible.y0 - clip.y0) * zoom)))
    canvas.copy(pix, pix.irect)
    return canvas


def _fit(width_pt: float, height_pt: float, max_side: int, min_side: int) -> tuple[float, int, int]:
    """Zoom and patch-aligned pixel size for a region whose longest side maps to ``max_side``."""
    limit = _floor_patch(max(1, int(max_side)))
    width_pt, height_pt = max(width_pt, 1e-3), max(height_pt, 1e-3)
    zoom = limit / max(width_pt, height_pt)
    lo = min(max(1, int(min_side)), limit)
    w = _ceil_patch(max(width_pt * zoom, lo), limit)
    h = _ceil_patch(max(height_pt * zoom, lo), limit)
    return zoom, w, h


# --------------------------------------------------------------------------- public api


def render_clip(page: pymupdf.Page, rect, max_side: int = 1536, min_side: int = 48, margin: float = 0.0) -> bytes:
    """PNG of ``rect`` (page points) grown by ``margin`` points on every side.

    The longest side is rendered at ``max_side`` px (rounded down to a multiple of
    48); both pixel dimensions are multiples of 48 and at least ``min_side``.
    """
    r = _as_rect(rect)
    if margin:
        r = pymupdf.Rect(r.x0 - margin, r.y0 - margin, r.x1 + margin, r.y1 + margin)
    zoom, w, h = _fit(r.width, r.height, max_side, min_side)
    return _render_exact(page, (r.tl + r.br) / 2, zoom, w, h).tobytes("png")


def render_page(page: pymupdf.Page, max_side: int = 1536) -> bytes:
    """PNG of the whole page, longest side ``max_side`` px (patch aligned)."""
    return render_clip(page, page.rect, max_side=max_side)


def _render_fit_cell(src, rect: pymupdf.Rect, width: int, height: int) -> pymupdf.Pixmap:
    zoom = min(width / max(rect.width, 1e-3), height / max(rect.height, 1e-3))
    return _render_exact(src, (rect.tl + rect.br) / 2, zoom, width, height)


def _fit_label(label: str, fontsize: float, width: float) -> str:
    text = label
    while text and pymupdf.get_text_length(text, fontname=_LABEL_FONT, fontsize=fontsize) > width:
        text = text[:-1]
    return text if text == label else (text[:-1] + "~" if len(text) > 1 else text)


def contact_sheet(
    page: pymupdf.Page,
    items: Sequence[tuple[str, pymupdf.Rect]],
    cell_px: int = 192,
    cols: int = 6,
    max_side: int = 1536,
    label_px: int = 24,
) -> list[tuple[bytes, list[str]]]:
    """Numbered contact sheets of page crops.

    Each item ``(label, rect)`` becomes one cell: a grey strip with the label on
    top of the crop (aspect preserved, widened to the cell with real context).
    Returns ``[(png, labels_on_this_sheet), ...]``; several sheets are produced
    when the items do not fit in ``max_side`` x ``max_side`` px.
    """
    if not items:
        return []
    limit = _floor_patch(max(1, int(max_side)))
    cell_px = max(PATCH, min(int(cell_px), limit))
    label_px = max(12, int(label_px))
    cell_h = cell_px + label_px
    cols = max(1, min(int(cols), limit // cell_px))
    rows_max = max(1, limit // cell_h)
    per_sheet = cols * rows_max
    fontsize = min(16.0, label_px * 0.68)
    dlist = page.get_displaylist()  # parse the page once for all crops (~17x faster on dense sheets)
    out: list[tuple[bytes, list[str]]] = []
    for start in range(0, len(items), per_sheet):
        chunk = list(items[start : start + per_sheet])
        ncols = min(cols, len(chunk))
        nrows = math.ceil(len(chunk) / ncols)
        width = _ceil_patch(ncols * cell_px, max(limit, ncols * cell_px))
        height = _ceil_patch(nrows * cell_h, max(limit, nrows * cell_h))
        tmp = pymupdf.open()
        try:
            sheet = tmp.new_page(width=width, height=height)
            sheet.draw_rect(sheet.rect, color=None, fill=(1, 1, 1))
            for i, (label, rect) in enumerate(chunk):
                r, c = divmod(i, ncols)
                x0, y0 = c * cell_px, r * cell_h
                strip = pymupdf.Rect(x0, y0, x0 + cell_px, y0 + label_px)
                sheet.draw_rect(strip, color=None, fill=(0.85, 0.85, 0.85))
                text = _fit_label(str(label), fontsize, cell_px - 8)
                sheet.insert_text((x0 + 4, y0 + label_px - (label_px - fontsize) / 2 - 2), text,
                                  fontname=_LABEL_FONT, fontsize=fontsize, color=(0, 0, 0))
                inner_w, inner_h = cell_px - 4, cell_h - label_px - 4
                pix = _render_fit_cell(dlist, _as_rect(rect), inner_w, inner_h)
                sheet.insert_image(pymupdf.Rect(x0 + 2, y0 + label_px + 2, x0 + 2 + inner_w, y0 + label_px + 2 + inner_h),
                                   pixmap=pix)
                sheet.draw_rect(pymupdf.Rect(x0 + 0.5, y0 + 0.5, x0 + cell_px - 0.5, y0 + cell_h - 0.5),
                                color=(0.45, 0.45, 0.45), width=1)
            png = sheet.get_pixmap(alpha=False, colorspace=pymupdf.csRGB).tobytes("png")
        finally:
            tmp.close()
        out.append((png, [str(lbl) for lbl, _ in chunk]))
    return out


def _starts(x0: float, length: float, tile: float, step: float) -> list[float]:
    if length <= tile + 1e-6:
        return [x0]
    n = int(math.ceil((length - tile) / max(step, 1e-6) - 1e-9)) + 1
    span = length - tile
    return [x0 + span * i / (n - 1) for i in range(n)]


def tiles(
    page: pymupdf.Page,
    tile_px: int = 1536,
    overlap: float = 0.15,
    dpi: float = DEFAULT_TILE_DPI,
    clip: Optional[pymupdf.Rect] = None,
) -> list[tuple[pymupdf.Rect, bytes]]:
    """Overlapping square tiles covering the page (or ``clip``) at a fixed ``dpi``.

    Tiles are ``tile_px`` px (patch aligned) and overlap by at least ``overlap``
    of their size; the last row/column is aligned to the region edge. Returns
    ``[(tile rect in page points, png), ...]`` row by row.
    """
    region = _as_rect(clip) & page.rect if clip is not None else pymupdf.Rect(page.rect)
    zoom = dpi / 72.0
    tile_px = _floor_patch(max(1, int(tile_px)))
    tile_pt = tile_px / zoom
    step = tile_pt * (1.0 - min(max(overlap, 0.0), 0.9))
    dlist = page.get_displaylist()
    out: list[tuple[pymupdf.Rect, bytes]] = []
    for y in _starts(region.y0, region.height, tile_pt, step):
        for x in _starts(region.x0, region.width, tile_pt, step):
            r = pymupdf.Rect(x, y, min(x + tile_pt, region.x1), min(y + tile_pt, region.y1))
            w = _ceil_patch(r.width * zoom, tile_px)
            h = _ceil_patch(r.height * zoom, tile_px)
            pix = _render_exact(dlist, (r.tl + r.br) / 2, zoom, w, h)
            out.append((r, pix.tobytes("png")))
    return out


def png_size(png: bytes) -> tuple[int, int]:
    """(width, height) of a PNG from its IHDR chunk."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
