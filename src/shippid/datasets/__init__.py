"""Stage 1: synthetic ship P&ID generation (readable vector PDFs + exact ground truth) and
collection of real drawings.

* :func:`generate` -- one synthetic drawing: ``(PIDDocument, DrawingSpec)``
* :func:`write_pdf` / :func:`to_bytes` -- render a :class:`DrawingSpec` as a readable PDF
* :func:`build_dataset` -- multiprocessing mass generation with manifest and splits
* :func:`scan_archive` / :func:`fetch_sources` -- real-drawing archive scan and downloads
"""

from .build import build_dataset
from .collect import fetch_sources, scan_archive
from .pdf_writer import DrawingSpec, render_png, to_bytes, write_pdf
from .synthetic import ALL_SYSTEMS, DrawingStyle, generate, random_style

__all__ = [
    "ALL_SYSTEMS", "DrawingSpec", "DrawingStyle", "build_dataset", "fetch_sources", "generate", "random_style",
    "render_png", "scan_archive", "to_bytes", "write_pdf",
]
